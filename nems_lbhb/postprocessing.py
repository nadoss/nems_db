import nems.analysis as na


def add_summary_statistics_by_condition(est,val,modelspec,evaluation_conditions,rec=None,**context):
    modelspec = na.api.standard_correlation_by_epochs(est,val,modelspec=modelspec,
            epochs_list=evaluation_conditions,rec=rec)
    return {'modelspec': modelspec}
