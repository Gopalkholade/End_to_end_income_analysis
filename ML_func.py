def cv_score_4ls(models=list,X=None,y=None,strat="mean"):
    """
    Models:the list of initialized models,
    X     :Complete Transformed Independent Variable,
    y     :Complete Transformed Dependent Variable,
    eval_strat:"mean","median","mode"
    """
    from tqdm import tqdm
    from sklearn.model_selection import cross_val_score
    model_scores={}
    for i in tqdm(models):
        score=cross_val_score(i,X,y,scoring="accuracy",n_jobs=5)
        if strat=="mean":
            model_scores[i]=score.mean()
        elif strat=="median":
            model_scores[i]=score.median()
        elif strat=="mode":
            model_scores[i]=score.mode()

    return model_scores

def seperate_by_type(df):
    """
    df: Any pandas Dataframe"""
    dim=df.select_dtypes(include="object").columns.to_list()
    mes=df.select_dtypes(exclude="object").columns.to_list()
    return dim,mes


