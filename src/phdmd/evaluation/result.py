import numpy as np
import os
import pickle
import pandas as pd

from phdmd.data.generate import sim
from phdmd.utils.modal_properties import compare_eigenmodes
from phdmd.utils.plotting import plot
from phdmd.utils.postprocess_data import compare_matrices


class Result:
    def __init__(self, lti_dict, exp, data_train, data_test, save_path, V=None):

        self.lti_dict = lti_dict
        self.exp = exp
        self.data_train = data_train
        self.data_test = data_test
        self.save_path = save_path

        if V is None:
            n, n_samples, n_u = data_train.shape
            self.V = np.eye(n)
            self.r = "full"
        else:
            self.V = V.to_numpy()
            # get reduced dimension
            if "POD" in lti_dict:
                self.pod_key = "POD"
            elif "ePOD" in lti_dict:
                self.pod_key = "ePOD"
            self.r = lti_dict[self.pod_key].order

        # initialize
        self.result_train_dict = {}
        self.result_test_dict = {}
        self.error_train_dict = {}
        self.error_test_dict = {}

    def get_all_results(self):

        self.simulate_identified_systems()
        self.calculate_error_values()
        self.compare_eigenmodes()
        self.plot_output()
        self.plot_error()

    def simulate_identified_systems(self):

        for key_method_name in self.lti_dict:
            # initial values
            if self.r == "full":
                x0_train = self.exp.x0
            else:
                x0_train = self.exp.x0_red

            # calculate training data
            U_id_train, X_id_train, Y_id_train = sim(
                self.lti_dict[key_method_name],
                self.exp.u,
                self.exp.T,
                x0=x0_train,
                method=self.exp.time_stepper,
            )

            self.result_train_dict[key_method_name] = {
                "U_id": U_id_train,
                "X_id": X_id_train,
                "Y_id": Y_id_train,
            }

            # initial values
            if self.r == "full":
                x0_test = self.exp.x0_test
            else:
                x0_test = self.exp.x0_test_red
            # calculate test data
            U_id_test, X_id_test, Y_id_test = sim(
                self.lti_dict[key_method_name],
                self.exp.u_test,
                self.exp.T_test,
                x0=x0_test,
                method=self.exp.time_stepper,
            )

            self.result_test_dict[key_method_name] = {
                "U_id": U_id_test,
                "X_id": X_id_test,
                "Y_id": Y_id_test,
            }

    def calculate_error_values(self):
        self.calculate_errors_from_result_dict(use_train_data=True)
        self.calculate_errors_from_result_dict(use_train_data=False)

    def calculate_errors_from_result_dict(self, use_train_data=False):
        if use_train_data:
            data = self.data_train.data
            result_dict = self.result_train_dict
            data_result_naming = "train"
        else:
            data = self.data_test.data
            result_dict = self.result_test_dict
            data_result_naming = "test"
        X_fom, Y_fom, U_fom = data[:3]

        error_mean_df = pd.DataFrame()
        for i_key, (key_method_name, results_one_method) in enumerate(
            result_dict.items()
        ):
            # reproject reduced state data
            X_reprojected = self.V.T @ results_one_method["X_id"]
            X_error_abs = np.abs(X_fom - X_reprojected)
            X_error_rel = (
                np.linalg.norm(X_error_abs, axis=0)
                / np.linalg.norm(X_fom, axis=0).mean()
            )
            # output error
            Y_error_abs = np.abs(Y_fom - results_one_method["Y_id"])
            Y_error_rel = Y_error_abs / Y_fom.mean()
            error_dict = {
                "X_error_abs": X_error_abs,
                "X_error_rel": X_error_rel,
                "Y_error_abs": Y_error_abs,
                "Y_error_rel": Y_error_rel,
                "method": key_method_name,
                # "r": r,
                # "noise": noise,
                "use_energy_weighted_POD": self.exp.use_energy_weighted_POD,
            }

            # calculate scalar mean error values
            # norm over outputs/nodes, mean over time
            X_error_abs_mean = np.linalg.norm(X_error_abs, axis=0).mean()
            X_error_rel_mean = np.linalg.norm(X_error_rel, axis=0).mean()
            Y_error_abs_mean = np.linalg.norm(Y_error_abs, axis=0).mean()
            Y_error_rel_mean = np.linalg.norm(Y_error_rel, axis=0).mean()
            error_mean = {
                "method": key_method_name,
                "X_error_abs_mean": X_error_abs_mean,
                "X_error_rel_mean": X_error_rel_mean,
                "Y_error_abs_mean": Y_error_abs_mean,
                "Y_error_rel_mean": Y_error_rel_mean,
            }
            if i_key == 0:
                # create
                error_mean_df = pd.DataFrame(error_mean, index=[0])
            else:
                # update
                error_mean_df = pd.concat(
                    [error_mean_df, pd.DataFrame(error_mean, index=[0])],
                    ignore_index=True,
                )

            if use_train_data:
                self.error_train_dict[key_method_name] = error_dict
            else:
                self.error_test_dict[key_method_name] = error_dict

        # save mean error
        add_plot_name = self.additional_plot_naming("all")
        csv_name = f"{self.exp.name}_{data_result_naming}_r{self.r}_error_mean{add_plot_name}.csv"
        csv_path = os.path.join(self.save_path, csv_name)
        error_mean_df.to_csv(csv_path)

    def compare_eigenmodes(self):

        for key_method_name, lti in self.lti_dict.items():
            if key_method_name == "POD" or key_method_name == "ePOD":
                # compare to instrusive POD -> skip POD-POD comparison
                continue

            if self.r == "full":
                comparison_lti = self.exp.fom
                compared_lti = "FOM"
            else:
                # compare MAC
                compared_lti = self.pod_key
                comparison_lti = self.lti_dict[self.pod_key]

            save_name_mac = os.path.join(
                self.save_path,
                f"MAC_r{self.r}_{compared_lti}_{key_method_name}.png",
            )
            title_name_mac = f"MAC_r_{self.r}_{compared_lti}_{key_method_name}"
            compare_eigenmodes(
                comparison_lti, lti, save_name=save_name_mac, title=title_name_mac
            )

    def plot_output(self):
        self.plot_output_from_result_dict(use_train_data=False)
        self.plot_output_from_result_dict(use_train_data=True)

    def plot_output_from_result_dict(self, use_train_data=False):
        if use_train_data:
            data = self.data_train.data
            result_dict = self.result_train_dict
            T = self.exp.T
            data_result_naming = "train"
        else:
            data = self.data_test.data
            result_dict = self.result_test_dict
            T = self.exp.T_test
            data_result_naming = "test"
        X_fom, Y_fom, U_fom = data[:3]

        Y_fom, _ = self.reshape_data_multi_scenario(T, Y_fom)

        n_y = Y_fom.shape[0]
        n_methods = len(result_dict.keys()) + 1  # + FOM
        n_t = Y_fom.shape[1]
        Y_all_methods = np.empty((n_methods, n_y, n_t))
        labels_all = np.empty((n_methods, n_y), dtype="object")

        # add FOM
        Y_all_methods[0] = Y_fom
        labels_all[0, :] = [f"FOM_y{i}" for i in np.arange(n_y) + 1]

        for i_key, (key_method_name, results_one_method) in enumerate(
            result_dict.items()
        ):

            Y_id = results_one_method["Y_id"]
            Y_id, _ = self.reshape_data_multi_scenario(T, Y_id)
            labels = []
            labels.extend(["y_fom"] * Y_fom.shape[0])
            labels.extend([f"y_{key_method_name}"] * Y_id.shape[0])
            labels = np.array(labels)[np.newaxis, :]

            Y = np.concatenate((Y_fom, Y_id), axis=0)

            # get naming from experiment
            add_plot_name = self.additional_plot_naming(key_method_name)

            ls = self.plot_style(Y.shape[0]).T
            # plot output of single method compared to fom
            plot(
                X=T,
                Y=Y,
                label=labels,
                ls=ls,
                xlabel="Time [s]",
                ylabel="Output",
                name=f"{self.exp.name}_{data_result_naming}_r{self.r}_output{add_plot_name}",
                subplots=False,
                save_path=self.save_path,
            )

            labels_all[i_key + 1] = [
                f"{key_method_name}_y{i}" for i in np.arange(n_y) + 1
            ]
            Y_all_methods[i_key + 1] = Y_id

        # plot outputs of all methods
        ls = self.plot_style(Y_all_methods.shape[0], Y_all_methods.shape[1])
        plot(
            X=np.tile(T, (n_methods, 1)),
            Y=Y_all_methods,
            label=labels_all,
            ls=ls,
            xlabel="Time [s]",
            ylabel="Output",
            name=f"{self.exp.name}_{data_result_naming}_r{self.r}_output_all{add_plot_name}",
            subplots=True,
            save_path=self.save_path,
        )

    @staticmethod
    def reshape_data_multi_scenario(T, XorY_error_data, labels=None):
        """
        resahpe data if n_t of Y is higher than t due to list of inputs u
        """
        if T.ndim == 1:
            lenght_T = len(T)
        elif T.ndim == 2:
            lenght_T = T.shape[1]
        data_shape_n_t = XorY_error_data.shape[-1]
        add_factor = data_shape_n_t / lenght_T
        if add_factor != 1:
            if add_factor.is_integer():
                # T = np.tile(T,(1,int(add_factor)))
                # add multiple scenario inputs to second to last axis
                if XorY_error_data.ndim == 1:
                    XorY_error_data = XorY_error_data[np.newaxis, :]
                data_shape = list(XorY_error_data.shape)
                data_shape_new = data_shape.copy()
                data_shape_new[-2] = data_shape[-2] * int(add_factor)
                data_shape_new[-1] = int(data_shape[-1] / add_factor)
                XorY_error_data = np.reshape(XorY_error_data, tuple(data_shape_new))
                # np.allclose(XorY_error_data_old[1,100:200],XorY_error_data[3,:]) -> True for old shape (2,200)
            else:
                raise ValueError(
                    f"add_factor needs to be an integer, but it's value is {add_factor}."
                )
        if labels is not None:
            if add_factor != 1:
                labels = np.tile(labels, (1, int(add_factor)))
        return XorY_error_data, labels

    def additional_plot_naming(self, method_name):
        if self.exp.perturb_energy_matrix:
            if isinstance(self.exp.perturb_value, str):
                add_perturb_name = f"_perturb_{self.exp.perturb_value}"
            else:
                add_perturb_name = f"_perturb{self.exp.perturb_value:.0e}"
        else:
            add_perturb_name = ""

        if self.exp.use_Riccatti_transform:
            add_Ricc_name = f"_Ricc{self.exp.use_Riccatti_transform}"
        else:
            add_Ricc_name = ""

        if self.exp.use_cholesky_like_fact:
            add_chol_fac = f"_Chol{self.exp.use_cholesky_like_fact}"
        else:
            add_chol_fac = ""

        if self.exp.use_cvx:
            add_cvx = f"_cvx{self.exp.use_cvx}_Jknown{self.exp.use_known_J}"
        else:
            add_cvx = ""

        if self.exp.use_projection_of_A:
            add_proj_A_snd = f"_projAsnd{self.exp.use_projection_of_A}"
        else:
            add_proj_A_snd = ""

        if self.r is not None:
            add_red_dim = f"_r{self.r}"
        else:
            add_red_dim = ""

        if self.exp.ordering != "JRH":
            add_ordering = f"_ordering{self.exp.ordering}"
        else:
            add_ordering = ""

        add_plot_name = f"_{method_name}_Bform{self.exp.use_Berlin}_init_{self.exp.HQ_init_strat}{add_perturb_name}{add_Ricc_name}{add_chol_fac}{add_cvx}{add_proj_A_snd}{add_red_dim}{add_ordering}"

        return add_plot_name

    def plot_style(self, n, k=1):
        """
        n: number of lines
        k: number of subplots
        """
        # https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
        ls_basic = np.array(
            [
                "solid",
                "dashed",
                "dashdot",
                (0, (5, 10)),  #'loosely dashed',
                (0, (3, 10, 1, 10)),  #'loosely dashdotted',
                (0, (3, 5, 1, 5, 1, 5)),  #'dashdotdotted',
                (0, (3, 10, 1, 10, 1, 10)),  #'loosely dashdotdotted'
            ],
            dtype="object",
        )[:, np.newaxis]

        if n > ls_basic.shape[0]:
            ls_basic = np.tile(ls_basic, (int(np.ceil(n / ls_basic.shape[0])), 1))

        ls = np.tile(ls_basic[:n], (1, k))

        return ls

    def plot_error(self):
        self.plot_error_from_error_dict(use_train_data=True)
        self.plot_error_from_error_dict(use_train_data=False)

    def plot_error_from_error_dict(self, use_train_data=False):
        if use_train_data:
            error_dict = self.error_train_dict
            T = self.exp.T
            data_result_naming = "train"
        else:
            error_dict = self.error_test_dict
            T = self.exp.T_test
            data_result_naming = "test"

        if "Original" in error_dict:
            # delete Original since error is zero
            error_dict.pop("Original", None)

        # initialize for methods
        n_x = error_dict[next(iter(error_dict))]["X_error_abs"].shape[0]
        n_y = error_dict[next(iter(error_dict))]["Y_error_abs"].shape[0]
        n_methods = len(error_dict.keys())
        n_t = error_dict[next(iter(error_dict))]["X_error_abs"].shape[1]
        X_error_abs_all_methods = np.empty((n_methods, n_x, n_t))
        X_error_rel_all_methods = np.empty((n_methods, n_x, n_t))
        Y_error_abs_all_methods = np.empty((n_methods, n_y, n_t))
        Y_error_rel_all_methods = np.empty((n_methods, n_y, n_t))
        labels_x_all = np.empty((n_methods, n_x), dtype="object")
        labels_y_all = np.empty((n_methods, n_y), dtype="object")

        for i_key, (key_method_name, error_one_method) in enumerate(error_dict.items()):

            # get error values from dict
            X_error_abs = error_one_method["X_error_abs"]
            X_error_rel = error_one_method["X_error_rel"]
            Y_error_abs = error_one_method["Y_error_abs"]
            Y_error_rel = error_one_method["Y_error_rel"]

            # plot differenty error values for each method separately
            self.plot_error_data(
                X_error_abs,
                T,
                key_method_name=key_method_name,
                data_type="state_abs",
                data_result_naming=data_result_naming,
            )
            self.plot_error_data(
                X_error_rel,
                T,
                key_method_name=key_method_name,
                data_type="state_rel",
                data_result_naming=data_result_naming,
            )
            self.plot_error_data(
                Y_error_abs,
                T,
                key_method_name=key_method_name,
                data_type="output_abs",
                data_result_naming=data_result_naming,
            )
            self.plot_error_data(
                Y_error_rel,
                T,
                key_method_name=key_method_name,
                data_type="output_rel",
                data_result_naming=data_result_naming,
            )

            labels_x_all[i_key] = [
                f"{key_method_name}_x{i}" for i in np.arange(n_x) + 1
            ]
            labels_y_all[i_key] = [
                f"{key_method_name}_y{i}" for i in np.arange(n_y) + 1
            ]
            X_error_abs_all_methods[i_key] = X_error_abs
            X_error_rel_all_methods[i_key] = X_error_rel
            Y_error_abs_all_methods[i_key] = Y_error_abs
            Y_error_rel_all_methods[i_key] = Y_error_rel

        # plot error of all methods
        T = np.tile(T, (n_methods, 1))
        for data, data_type, labels in zip(
            [
                X_error_abs_all_methods,
                X_error_rel_all_methods,
                Y_error_abs_all_methods,
                Y_error_rel_all_methods,
            ],
            ["state_abs", "state_rel", "output_abs", "output_rel"],
            [labels_x_all, labels_x_all, labels_y_all, labels_y_all],
        ):
            data, labels = self.cut_data(data, labels)
            self.plot_error_data(
                data,
                T,
                key_method_name=key_method_name,
                data_type=data_type,
                data_result_naming=data_result_naming,
                all=True,
                labels=labels,
            )
        # self.plot_error_data(X_error_abs_all_methods,T,key_method_name=key_method_name,data_type="state_abs",data_result_naming=data_result_naming,all=True,labels=labels_x_all)
        # self.plot_error_data(X_error_rel_all_methods,T,key_method_name=key_method_name,data_type="state_rel",data_result_naming=data_result_naming,all=True,labels=labels_x_all)
        # self.plot_error_data(Y_error_abs_all_methods,T,key_method_name=key_method_name,data_type="output_abs",data_result_naming=data_result_naming,all=True,labels=labels_y_all)
        # self.plot_error_data(Y_error_rel_all_methods,T,key_method_name=key_method_name,data_type="output_rel",data_result_naming=data_result_naming,all=True,labels=labels_y_all)

    def cut_data(self, XorY_error_data, labels=None):
        # cut data to most significant ones
        if XorY_error_data.ndim == 3:
            if XorY_error_data.shape[1] > 6:
                sorted_indices = np.argsort(XorY_error_data, axis=1)
                sorted_indices = sorted_indices[
                    0, :6, -1
                ]  # first method, last time step
                XorY_error_data = XorY_error_data[:, sorted_indices]
                if labels is not None:
                    labels = labels[:, sorted_indices]
            else:
                # keep data as it is
                pass
        else:
            raise NotImplementedError("Check dimension")
        return XorY_error_data, labels

    def plot_error_data(
        self,
        XorY_error_data,
        T,
        key_method_name,
        data_type="state_abs",
        data_result_naming="testing",
        all=False,
        labels=None,
    ):
        """
        data_type (str): state_rel | state_abs | output_rel | output_abs
        """
        assert data_type in ["state_rel", "state_abs", "output_rel", "output_abs"]
        if data_type == "state_rel":
            label_name = "e_x"
            ylabel_name = "rel. state error"
        elif data_type == "state_abs":
            label_name = "e_x"
            ylabel_name = "abs. state error"
        elif data_type == "output_rel":
            label_name = "e_y"
            ylabel_name = "rel. output error"
        elif data_type == "output_abs":
            label_name = "e_y"
            ylabel_name = "abs. output error"

        XorY_error_data, labels = self.reshape_data_multi_scenario(
            T, XorY_error_data, labels=labels
        )

        if all:
            # if all methods are used
            all_name = "all_"
            ls = self.plot_style(XorY_error_data.shape[0], XorY_error_data.shape[1])
            subplots = True
        else:
            # single method
            all_name = ""
            ls = self.plot_style(XorY_error_data.shape[0]).T
            subplots = False

        # cut data to most significant ones
        if XorY_error_data.ndim == 1:
            # only one time trajectory
            label_indices = [f"{label_name}1"]
        else:
            if XorY_error_data.ndim == 2:
                significant_axis = 0
            elif XorY_error_data.ndim == 3:
                significant_axis = 1
            if XorY_error_data.shape[significant_axis] > 6:
                sorted_indices = np.argsort(XorY_error_data, axis=significant_axis)
                if XorY_error_data.ndim == 1:
                    sorted_indices = sorted_indices[:6]
                elif XorY_error_data.ndim == 2:
                    sorted_indices = sorted_indices[:6, -1]  # maximum at last time step
                elif XorY_error_data.ndim == 3:
                    sorted_indices = sorted_indices[
                        0, :6, -1
                    ]  # first data point, last time step sorting
                if significant_axis == 0:
                    XorY_error_data = XorY_error_data[sorted_indices]
                elif significant_axis == 1:
                    XorY_error_data = XorY_error_data[:, sorted_indices]
                label_indices = sorted_indices + 1
            else:
                label_indices = np.arange(XorY_error_data.shape[0]) + 1

        if labels is None:
            labels = [f"{label_name}{i}" for i in label_indices]

        add_plot_name = self.additional_plot_naming(key_method_name)
        # if XorY_error_data.ndim == 1:
        #     data_shape_n_t = XorY_error_data.shape[0]
        # elif XorY_error_data.ndim == 2:
        #     data_shape_n_t = XorY_error_data.shape[1]
        # elif XorY_error_data.ndim == 3:
        # data_shape_n_t = XorY_error_data.shape[-1]

        plot(
            X=T,
            Y=XorY_error_data,
            label=labels,
            ls=ls,
            xlabel="Time [s]",
            ylabel=ylabel_name,
            name=f"{self.exp.name}_{data_result_naming}_r{self.r}_error_{all_name}{data_type}{add_plot_name}",
            subplots=subplots,
            save_path=self.save_path,
        )

    def compare_matrices(self, lti, latex_output=False):
        compare_matrices(self.exp, lti, latex_output=latex_output)

    def plot_guitar_plots(self):
        """
        manual plot of the guitar for the statusseminar
        """
        data = self.data_test.data
        result_dict = self.result_test_dict
        # T = self.exp.T_test
        data_result_naming = "test"

        method_name = "pHDMD"
        X_fom, Y_fom, U_fom = data[:3]
        Y_id = result_dict[method_name]["Y_id"]
        T = np.arange(Y_fom.shape[1])
        Y = np.concatenate((Y_fom, Y_id), axis=0)

        ls = self.plot_style(Y.shape[0]).T
        if self.exp.use_energy_weighted_POD:
            # energy POD
            POD_variant = "ePOD"
        else:
            # normal POD
            POD_variant = "POD"
        labels = np.array(["y_fom", f"y_{POD_variant}"])[np.newaxis, :]
        add_plot_name = self.additional_plot_naming(method_name)

        plot(
            X=T,
            Y=Y,
            label=labels,
            ls=ls,
            xlabel="Time [s]",
            ylabel="Output",
            name=f"{self.exp.name}_{data_result_naming}_{POD_variant}_r{self.r}_output_guitar{add_plot_name}",
            subplots=False,
            save_path=self.save_path,
        )

    def save_instance(self, filename, result_instance):
        with open(f"{filename}.pickle", "wb") as file_:
            pickle.dump(result_instance, file_)

    @classmethod
    def load_instance(self, filename):
        return pickle.load(open(f"{filename}.pickle", "rb", -1))

    def save(obj, filename="result_instance"):
        # with open(f"{filename}.pickle", "wb") as file_:
        #     pickle.dump((obj.__class__, obj.__dict__), file_)
        data = np.array([(obj.__class__, obj.__dict__)], dtype="object")
        np.save(f"{filename}.npy", data, allow_pickle=True)
        return

    @staticmethod
    def restore_static(cls, attributes):
        obj = cls.__new__(cls)
        obj.__dict__.update(attributes)
        return obj

    @classmethod
    def restore_class(cls, attributes):
        obj = cls.__new__(cls)
        obj.__dict__.update(attributes)
        return obj

    def restore(cls, attributes):
        obj = cls.__new__(cls)
        obj.__dict__.update(attributes)
        return obj

        Y_fom, _ = self.reshape_data_multi_scenario(T, Y_fom)

        n_y = Y_fom.shape[0]
        n_methods = len(result_dict.keys()) + 1  # + FOM
        n_t = Y_fom.shape[1]
        Y_all_methods = np.empty((n_methods, n_y, n_t))
        labels_all = np.empty((n_methods, n_y), dtype="object")

        # add FOM
        Y_all_methods[0] = Y_fom
        labels_all[0, :] = [f"FOM_y{i}" for i in np.arange(n_y) + 1]

        for i_key, (key_method_name, results_one_method) in enumerate(
            result_dict.items()
        ):

            Y_id = results_one_method["Y_id"]
            Y_id, _ = self.reshape_data_multi_scenario(T, Y_id)
            labels = []
            labels.extend(["y_fom"] * Y_fom.shape[0])
            labels.extend([f"y_{key_method_name}"] * Y_id.shape[0])
            labels = np.array(labels)[np.newaxis, :]

            Y = np.concatenate((Y_fom, Y_id), axis=0)

            # get naming from experiment
            add_plot_name = self.additional_plot_naming(key_method_name)

            ls = self.plot_style(Y.shape[0]).T
            # plot output of single method compared to fom
            plot(
                X=T,
                Y=Y,
                label=labels,
                ls=ls,
                xlabel="Time [s]",
                ylabel="Output",
                name=f"{self.exp.name}_{data_result_naming}_r{self.r}_output{add_plot_name}",
                subplots=False,
                save_path=self.save_path,
            )

            labels_all[i_key + 1] = [
                f"{key_method_name}_y{i}" for i in np.arange(n_y) + 1
            ]
            Y_all_methods[i_key + 1] = Y_id

        # plot outputs of all methods
        ls = self.plot_style(Y_all_methods.shape[0], Y_all_methods.shape[1])
