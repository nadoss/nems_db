import nems.analysis as na


def add_summary_statistics_by_condition(est,val,modelspecs,rec,evaluation_conditions,**context):
    modelspecs = na.api.standard_correlation_by_epochs(est,val,modelspecs,
            evaluation_conditions,rec=rec)
    return {'modelspecs': modelspecs}
