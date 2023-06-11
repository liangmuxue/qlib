# 分类定义,根据价格幅度区间进行分类
CLASS_VALUES={1:[-1000,-10],2:[-10,-9],3:[-9,-8],4:[-8,-7],5:[-7,-6],6:[-6,-5],7:[-5,-4],8:[-4,-3],9:[-3,-2],10:[-2,-1],11:[-1,0]
              ,12:[0,1],13:[1,2],14:[2,3],15:[3,4],16:[4,5],17:[5,6],18:[6,7],19:[7,8],20:[8,9],21:[9,10],22:[10,1000]}

# 涨跌幅分类区间,包括区间以及损失权重
# CLASS_SIMPLE_VALUES={0:[[-1000,-5],0.3],1:[[-5,0],0.1],2:[[0,3],0.2],3:[[3,1000],0.4]}
CLASS_SIMPLE_VALUES={0:[[-1000,-5],0.3],1:[[-5,0],0.1],2:[[0,5],0.2],3:[[5,1000],0.4]}
CLASS_SIMPLE_VALUE_MAX = list(CLASS_SIMPLE_VALUES)[-1]
CLASS_SIMPLE_VALUE_SEC = list(CLASS_SIMPLE_VALUES)[-2]

def get_simple_class_weight():
    return [v[1] for k, v in CLASS_SIMPLE_VALUES.items()]

def get_simple_class(target_value):
    result = None
    index = 0
    for k, v in CLASS_SIMPLE_VALUES.items():
        index += 1
        if (target_value>=v[0][0] and target_value<v[0][1]):
            result = k
            break
    # 保证不超出最大最小范围
    if result is None:
        if target_value<0:
            result = 0
        else:
            result = CLASS_SIMPLE_VALUE_MAX
    return result
       
VALUE_BINS = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0, 1, 2, 3,4, 5, 6, 7,8, 9, 10]
#CLASS_VALUES={0:[-1000,-10],1:[-10,-5],2:[-5,0],3:[0,5],4:[5,10],5:[10,1000]}

PAD, PAD_IDX = "<PAD>", 0 # padding
SOS, SOS_IDX = "<SOS>", 16 # start of sequence
WEOS, WEOS_IDX = "<EOS>", 23 # end of sequence
EOS, EOS_IDX = "<EOS>", 17 # end of sequence
UNK, UNK_IDX = "<UNK>", 3 # unknown token

# 均线斜率分类,包括上升，下降，平稳，震荡
SLOPE_SHAPE_RAISE = 0
SLOPE_SHAPE_FALL = 1
SLOPE_SHAPE_SMOOTH = 2
SLOPE_SHAPE_SHAKE = 3