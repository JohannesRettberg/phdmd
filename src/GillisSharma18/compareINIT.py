from stablePassiveFGM import stablePassiveFGM

def compareINIT(sys, options):
    n = sys['A'].shape[0]
    m = sys['B'].shape[1]

    # Standard initialization
    print('*********************************')
    print('1) Standard initialization:')
    print('*********************************')
    PHforms, es, ts = stablePassiveFGM(sys, options)
    print(f'Standard init., error = {es[-1]:.2f}.')

    # LMIs + formula
    print('*********************************')
    print('2) LMIs + formula initialization:')
    print('*********************************')
    options['init'] = 2
    PHforml, el, tl = stablePassiveFGM(sys, options)
    print(f'LMIs + formula, error = {el[-1]:.2f}.')

    # LMIs + solve
    print('*********************************')
    print('3) LMIs + solve initialization:')
    print('*********************************')
    options['init'] = 3
    PHformo, eo, to = stablePassiveFGM(sys, options)
    print(f'LMIs + solve, error = {eo[-1]:.2f}.')

    return PHforms, PHforml, PHformo, es, ts, el, tl, eo, to