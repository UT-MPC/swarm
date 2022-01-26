HETERO_STRATEGY = ['DROppCL test', 'baseline', 'dropout', 'quantize', 'DROppCL']

def is_hetero_strategy(strategy):
    return strategy in HETERO_STRATEGY