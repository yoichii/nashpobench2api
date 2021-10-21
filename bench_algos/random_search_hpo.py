# Random Search for Hyper-Parameter Optimization, JMLR 2012
def do_hpo(args, api, cellcode):
    while api.get_total_cost() < args.time_budget_hpo:
        # get key(=cellcode, lr, batch_size)
        key = api.get_random_key()
        key['cellcode'] = cellcode
        # query accuracy
        acc, cost = api.query_by_key(**key, epoch=12)
    return api.get_results(epoch=args.test_epoch)
