from qlib.contrib.data.handler import DataHandlerLP,check_transform_proc,_DEFAULT_LEARN_PROCESSORS
from qlib.data.dataset.loader import QlibDataLoader
from qlib.data.ops import Mad
from .factor_ext import build_rsi_factor_str,build_rvi_factor_str,build_aos_factor_str

class TftDataHandler(DataHandlerLP):
    """负责从底层取得原始数据"""
    
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processor=None,
        **kwargs
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": kwargs.get("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processor": inst_processor,
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            init_data=kwargs["init_data"]
        )

    def get_feature_config(self):
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                # "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
                "feature": ["OPEN", "HIGH", "LOW","CLOSE"],
            },
            "volume": {},
            "turnover": {},
            "rolling": {},
            # 添加nan_validate_label，用于数据空值校验
            "validate_fields": {},
        }
        return self.parse_config_to_fields(conf)
    
    def get_label_config(self):
        return (["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"])
    
    @staticmethod
    def parse_config_to_fields(config):
        """create factors from config
    
        config = {
            'kbar': {}, # whether to use some hard-code kbar features
            'price': { # whether to use raw price features
                'windows': [0, 1, 2, 3, 4], # use price at n days ago
                'feature': ['OPEN', 'HIGH', 'LOW'] # which price field to use
            },
            'volume': { # whether to use raw volume features
                'windows': [0, 1, 2, 3, 4], # use volume at n days ago
            },
            'rolling': { # whether to use rolling operator based features
                'windows': [5, 10, 20, 30, 60], # rolling windows size
                'include': ['ROC', 'MA', 'STD'], # rolling operator to use
                #if include is None we will use default operators
                'exclude': ['RANK'], # rolling operator not to use
            }
        }
        """
        fields = []
        names = []
        # 校验无效值
        if "validate_fields" in config:
            fields += ["$close"]
            names += ["value_validate"]
        if "kbar" in config:
            fields += [
                "($close-$open)/$open",
                "($high-$low)/$open",
                "($close-$open)/($high-$low+1e-12)",
                "($high-Greater($open, $close))/$open",
                "($high-Greater($open, $close))/($high-$low+1e-12)",
                "(Less($open, $close)-$low)/$open",
                "(Less($open, $close)-$low)/($high-$low+1e-12)",
                "(2*$close-$high-$low)/$open",
                "(2*$close-$high-$low)/($high-$low+1e-12)",
                "EMA($close, 12) - EMA($close, 26)",
                "EMA((EMA($close, 12) - EMA($close, 26)), 9)",
                "2*(EMA($close, 12) - EMA($close, 26)-EMA((EMA($close, 12) - EMA($close, 26)), 9))",
                "$close/Ref($close,5)*100",
                build_rvi_factor_str(),
                build_aos_factor_str(),
                "($high - EMA($close, 12))/EMA($close, 12)",
            ]
            names += [
                "KMID",
                "KLEN",
                "KMID2",
                "KUP",
                "KUP2",
                "KLOW",
                "KLOW2",
                "KSFT",
                "KSFT2",
                "DIFF",
                "DEA",
                "MACD",
                "MOMENTUM", # 动量指数
                "RVI", # 相对活力指数
                "AOS", # 加速振荡器
                "BULLS", # 牛市力度指数
            ]
        if "price" in config:
            windows = config["price"].get("windows", range(5))
            feature = config["price"].get("feature", ["OPEN", "HIGH", "LOW", "CLOSE"])
            # 添加标准指标开盘收盘等
            for field in feature:
                field = field.lower()
                fields += ["${}".format(field)]
                names += [field.upper()]                
            for field in feature:
                field = field.lower()
                fields += ["Ref($%s, %d)/$close" % (field, d) if d != 0 else "$%s/$close" % field for d in windows]
                names += [field.upper() + str(d) for d in windows]
            fields += ["($close/Ref($close, 1)-1)*100"]
            names += ["PRICE_SCOPE"]    
        if "volume" in config:
            windows = config["volume"].get("windows", range(5))
            fields += ["Ref($volume, %d)/$volume" % d if d != 0 else "$volume/$volume" for d in windows]
            names += ["VOLUME" + str(d) for d in windows]
            fields += ["$volume"]
            names += ["VOLUME_CLOSE"]
        if "turnover" in config:
            windows = config["turnover"].get("windows", range(5))
            fields += ["Ref($turnover, %d)/$turnover" % d if d != 0 else "$turnover/$turnover" for d in windows]
            names += ["TURNOVER" + str(d) for d in windows]
            fields += ["$turnover"]
            names += ["TURNOVER_CLOSE"]            
        if "rolling" in config:
            windows = config["rolling"].get("windows", [5, 10, 20, 30, 60])
            include = config["rolling"].get("include", None)
            exclude = config["rolling"].get("exclude", [])
            # `exclude` in dataset config unnecessary filed
            # `include` in dataset config necessary field
            use = lambda x: x not in exclude and (include is None or x in include)
            if use("ROC"):
                fields += ["Ref($close, %d)/$close" % d for d in windows]
                names += ["ROC%d" % d for d in windows]
            if use("MA"):
                fields += ["Mean($close, %d)/$close" % d for d in windows]
                names += ["MA%d" % d for d in windows]
            if use("MASCOPE"):
                fields += ["((Mean($close,{})/$close)/Ref(Mean($close, {})/$close,1)-1)*100".format(d,d) for d in windows]
                names += ["MASCOPE%d" % d for d in windows]                
            if use("STD"):
                fields += ["Std($close, %d)/$close" % d for d in windows]
                names += ["STD%d" % d for d in windows]
            if use("BETA"):
                fields += ["Slope($close, %d)/$close" % d for d in windows]
                names += ["BETA%d" % d for d in windows]
            if use("RSQR"):
                fields += ["Rsquare($close, %d)" % d for d in windows]
                names += ["RSQR%d" % d for d in windows]
            if use("RESI"):
                fields += ["Resi($close, %d)/$close" % d for d in windows]
                names += ["RESI%d" % d for d in windows]
            if use("MAX"):
                fields += ["Max($high, %d)/$close" % d for d in windows]
                names += ["MAX%d" % d for d in windows]
            if use("LOW"):
                fields += ["Min($low, %d)/$close" % d for d in windows]
                names += ["MIN%d" % d for d in windows]
            if use("QTLU"):
                fields += ["100*Quantile($close, %d, 0.8)/$close - 100" % d for d in windows]
                names += ["QTLU%d" % d for d in windows]
            if use("HIGH_QTLU"):
                fields += ["Quantile($high, %d, 0.8)/$high" % d for d in windows]
                names += ["HIGH_QTLU%d" % d for d in windows]               
            if use("QTLUMA"):
                fields += ["Mean(100*Quantile($close, %d, 0.8)/$close-100,%d)" % (d,d) for d in windows]
                names += ["QTLUMA%d" % d for d in windows]                
            if use("QTLD"):
                fields += ["Quantile($close, %d, 0.2)/$close" % d for d in windows]
                names += ["QTLD%d" % d for d in windows]
            if use("RANK"):
                fields += ["Rank($close, %d)" % d for d in windows]
                names += ["RANK%d" % d for d in windows]
            if use("RANKMA"):
                fields += ["Mean(Rank($close, %d),%d)" % (d, d) for d in windows]
                names += ["RANKMA%d" % d for d in windows]                
            if use("RSV"):
                fields += ["($close-Min($low, %d))/(Max($high, %d)-Min($low, %d)+1e-12)" % (d, d, d) for d in windows]
                names += ["RSV%d" % d for d in windows]
            if use("IMAX"):
                fields += ["IdxMax($high, %d)/%d" % (d, d) for d in windows]
                names += ["IMAX%d" % d for d in windows]
            if use("IMIN"):
                fields += ["IdxMin($low, %d)/%d" % (d, d) for d in windows]
                names += ["IMIN%d" % d for d in windows]
            if use("IMXD"):
                fields += ["(IdxMax($high, %d)-IdxMin($low, %d))/%d" % (d, d, d) for d in windows]
                names += ["IMXD%d" % d for d in windows]
            if use("CORR"):
                fields += ["Corr($close, Log($volume+1), %d)" % d for d in windows]
                names += ["CORR%d" % d for d in windows]
            if use("CORD"):
                fields += ["Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), %d)" % d for d in windows]
                names += ["CORD%d" % d for d in windows]
            if use("CNTP"):
                fields += ["Mean($close>Ref($close, 1), %d)" % d for d in windows]
                names += ["CNTP%d" % d for d in windows]
            if use("CNTN"):
                fields += ["Mean($close<Ref($close, 1), %d)" % d for d in windows]
                names += ["CNTN%d" % d for d in windows]
            if use("KURT"):
                fields += ["Kurt($close, %d)" % d for d in windows]
                names += ["KURT%d" % d for d in windows]           
            if use("SKEW"):
                fields += ["Skew($close, %d)" % d for d in windows]
                names += ["SKEW%d" % d for d in windows]                          
            if use("CNTD"):
                fields += ["Mean($close>Ref($close, 1), %d)-Mean($close<Ref($close, 1), %d)" % (d, d) for d in windows]
                names += ["CNTD%d" % d for d in windows]
            if use("SUMP"):
                fields += [
                    "Sum(Greater($close-Ref($close, 1), 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d)
                    for d in windows
                ]
                names += ["SUMP%d" % d for d in windows]
            if use("SUMPMA"):
                fields += [
                    "Mean(Sum(Greater($close-Ref($close, 1), 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12),%d)" % (d, d,d)
                    for d in windows
                ]
                names += ["SUMPMA%d" % d for d in windows]                
            if use("SUMN"):
                fields += [
                    "Sum(Greater(Ref($close, 1)-$close, 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d)
                    for d in windows
                ]
                names += ["SUMN%d" % d for d in windows]
            if use("SUMD"):
                fields += [
                    "(Sum(Greater($close-Ref($close, 1), 0), %d)-Sum(Greater(Ref($close, 1)-$close, 0), %d))"
                    "/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d, d)
                    for d in windows
                ]
                names += ["SUMD%d" % d for d in windows]
            if use("VMA"):
                fields += ["Mean($volume, %d)/($volume+1e-12)" % d for d in windows]
                names += ["VMA%d" % d for d in windows]
            if use("VSTD"):
                fields += ["Std($volume, %d)/($volume+1e-12)" % d for d in windows]
                names += ["VSTD%d" % d for d in windows]
            if use("WVMA"):
                # 添加小数，避免NAN
                fields += [
                    "Std(Abs($close/Ref($close, 1)-1+1e-5)*$volume, %d)/(Mean(Abs($close/Ref($close, 1)-1+1e-5)*$volume, %d)+1e-12)"
                    % (d, d)
                    for d in windows
                ]
                names += ["WVMA%d" % d for d in windows]
            if use("VSUMP"):
                fields += [
                    "Sum(Greater($volume-Ref($volume, 1), 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)"
                    % (d, d)
                    for d in windows
                ]
                names += ["VSUMP%d" % d for d in windows]
            if use("VSUMN"):
                fields += [
                    "Sum(Greater(Ref($volume, 1)-$volume, 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)"
                    % (d, d)
                    for d in windows
                ]
                names += ["VSUMN%d" % d for d in windows]
            if use("VSUMD"):
                fields += [
                    "(Sum(Greater($volume-Ref($volume, 1), 0), %d)-Sum(Greater(Ref($volume, 1)-$volume, 0), %d))"
                    "/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)" % (d, d, d)
                    for d in windows
                ]
                names += ["VSUMD%d" % d for d in windows]
            if use("TSTD"):
                # 对换手率进行STD
                fields += ["Std($turnover, %d)/($turnover+1e-12)" % d for d in windows]
                names += ["TSTD%d" % d for d in windows]
            if use("TMA"):
                # 对换手率进行MEAN
                fields += ["Mean($turnover, %d)/($turnover+1e-12)" % d for d in windows]
                names += ["TMA%d" % d for d in windows]                
            if use("TSUMP"):
                # 对换手率进行SUM
                fields += [
                    "Sum(Greater($turnover-Ref($turnover, 1), 0), %d)/(Sum(Abs($turnover-Ref($turnover, 1)), %d)+1e-12)"
                    % (d, d)
                    for d in windows
                ]
                names += ["TSUMP%d" % d for d in windows]       
            if use("RSI"):
                # 自定义rsi指标
                fields += [build_rsi_factor_str(d) for d in windows]
                names += ["RSI%d" % d for d in windows]      
            if use("REV"):
                # 自定义动量反转指标
                fields += ["Sum(($volume*$close/Ref($close, 1)),%d)" % d for d in windows]
                names += ["REV%d" % d for d in windows]         
            if use("WR"):
                # 自定义威廉指数
                fields += ["(Max($HIGH,{})-$close)/(Max($HIGH,{})-Min($LOW,{}))*100".format(d,d,d) for d in windows]
                names += ["WR%d" % d for d in windows]      
            if use("CCI"):
                # 自定义顺势指标
                for d in windows:
                    tp_str = "($HIGH+$LOW+$close)/3"
                    ma_str = "Mean(({}),{})".format(tp_str,d)
                    md_str = "Mean(Abs({}-{}),{})".format(ma_str,tp_str,d)
                    field_combine = "({}-{})/{}/0.015".format(tp_str,ma_str,md_str)   
                    fields += [field_combine]              
                names += ["CCI%d" % d for d in windows]                 
            if use("OBV"):
                # 自定义能量潮指标
                fields += ["Mean((($close-$LOW) - ($HIGH-$close))/($HIGH-$LOW)*$volume,{})".format(d) for d in windows]
                names += ["OBV%d" % d for d in windows]                                        
        return fields, names
    
    
class FuturesDataHandler(TftDataHandler):    

    def get_feature_config(self):
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                # "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
                "feature": ["OPEN", "HIGH", "LOW","CLOSE","REFCLOSE","HOLD"],
            },
            "volume": {},
            "turnover": {},
            "rolling": {},
            # 添加nan_validate_label，用于数据空值校验
            "validate_fields": {},
        }
        return self.parse_config_to_fields(conf)    
    
    