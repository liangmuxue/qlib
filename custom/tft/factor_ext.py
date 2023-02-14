def build_rsi_factor_str(days):
    raise_template = "If(Gt($close,Ref($close, 1)), ($close-Ref($close, 1))/Ref($close, 1), 0)"
    down_template = "If(Le($close,Ref($close, 1)), (Ref($close, 1)-$close)/Ref($close, 1),0)"
    mean_raise_template = "Mean(({}),{})".format(raise_template,days)
    mean_down_template = "Mean(({}),{})".format(down_template,days)
    total_template = "100*{}/({}+{})".format(mean_raise_template,mean_down_template,mean_raise_template)
    
    return total_template