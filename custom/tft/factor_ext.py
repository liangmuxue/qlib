def build_rsi_factor_str(days):
    """构建RSI指标"""
    raise_template = "If(Gt($close,Ref($close, 1)), ($close-Ref($close, 1))/Ref($close, 1), 0)"
    down_template = "If(Le($close,Ref($close, 1)), (Ref($close, 1)-$close)/Ref($close, 1),0)"
    mean_raise_template = "Mean(({}),{})".format(raise_template,days)
    mean_down_template = "Mean(({}),{})".format(down_template,days)
    total_template = "{}/({}+{})".format(mean_raise_template,mean_down_template,mean_raise_template)
    
    return total_template

def build_rev_factor_str(days):
    """构建动量反转指标"""
    
    item_template = "$volume*$close/Ref($close, 1)"
    total_template = "Sum({})".format(item_template,days)
    return total_template

def build_aos_factor_str(days=4):
    """构建加速振荡器指标"""
    
    md_price_template = "($high+$low)/2"
    ao_str = "Mean({},5) - Mean({},34)".format(md_price_template,md_price_template)
    ac_str = "{} - Mean({},5)".format(ao_str,ao_str)
    return ac_str

def build_rvi_factor_str(days=4):
    """构建相对活力指数"""
    
    move_avg_template = "($close-$open) + 2*(Ref($close, 1)-Ref($open, 1)) + 2*(Ref($close, 2)-Ref($open, 2)) + (Ref($close, 3)-Ref($open, 3))"
    range_avg_template = "($high-$low) + 2*(Ref($high, 1)-Ref($low, 1)) + 2*(Ref($high, 2)-Ref($low, 2)) + (Ref($high, 3)-Ref($low, 3))"
    ma_str = "Sum({},{})".format(move_avg_template,days)
    ra_str = "Sum({},{})".format(range_avg_template,days)
    rvi_str = "{}/{}".format(ma_str,ra_str)   
    return rvi_str


