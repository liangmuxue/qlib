from enum import Enum, unique

from .busi_struct import BasePersistence
from workflow.constants_enum import WorkflowStatus,WorkflowSubStatus,FrequencyType
from persistence.common_dict import CommonDictEnum
    
class WfTaskStore(BasePersistence):
    """工作流数据持久化处理"""
    
    def __init__(self):
        
        super().__init__()

    def get_workflow_id_by_name(self,workflow_name):
        row = self.dbaccessor.do_query("select id from workflow where name='{}'".format(workflow_name))[0]
        return row[0]         
        
    def create_workflow_task(self,workflow_id):
        """生成工作流数据 """

        task_batch = self.dbaccessor.do_query("select max(task_batch) from workflow_task".format(workflow_id))[0][0]
        if task_batch is None:
            task_batch = 1
        else:
            task_batch = task_batch + 1
        insert_sql = "insert into workflow_task(workflow_id,task_batch,status) values(%s,%s,%s)" 
        self.dbaccessor.do_inserto_withparams(insert_sql,(workflow_id,task_batch,WorkflowStatus.created.value))
        return self.get_workflow_task(task_batch)

    def create_workflow_task_calendar(self,task_id,batch_list):
        """生成工作流日历 """

        # 删除之前的重复日历
        self.dbaccessor.do_inserto("delete from workflow_calendar where task_id={}".format(task_id))
        for index,batch_item in enumerate(batch_list):
            sequence = index + 1
            for sub_seq,item in enumerate(batch_item):
                insert_sql = "insert into workflow_calendar(task_id,sequence,sub_seq,working_day,type) values(%s,%s,%s,%s,%s)" 
                self.dbaccessor.do_inserto_withparams(insert_sql,(task_id,sequence,sub_seq+1,item,FrequencyType.WEEK.value))
 
    def get_workflow_task_firstday_calendar(self,task_id,working_day):
        """获取工作日历数据，只匹配批次内首日日历"""

        rows = self.dbaccessor.do_query("select id,sequence from workflow_calendar where task_id={} and type={} and " \
                                       "working_day={} and sub_seq=1".format(task_id,FrequencyType.WEEK.value,working_day))
        if len(rows)==0:
            return None
        row = rows[0]                               
        rtn_obj = {"id":row[0],"sequence":row[1]}
        return rtn_obj 
 
    def get_workflow_task_calendar(self,task_id,sequence):
        """根据序号，取得对应的批次日历"""

        rows = self.dbaccessor.do_query("select id,sequence,working_day from workflow_calendar where task_id={} and type={} and " \
                                       "sequence={}".format(task_id,FrequencyType.WEEK.value,sequence))
        return [row[2] for row in rows] 
                                     
    def get_workflow_task(self,task_batch):
        """获取工作流数据 """

        row = self.dbaccessor.do_query("select id,workflow_id,task_batch,status,cur_working_day,cur_task_detail_id from workflow_task where task_batch={}".format(task_batch))[0]
        rtn_obj = {"id":row[0],"workflow_id":row[1],"task_batch":row[2],"status":row[3],"cur_working_day":row[4],"cur_task_detail_id":row[5]}
        return rtn_obj 
    
    def update_workflow_task_status(self,task_batch,status):
        """修改工作流任务状态 """
        
        update_sql = "update workflow_task set status=%s where task_batch=%s"
        self.dbaccessor.do_inserto_withparams(update_sql,(status,task_batch))

    def update_workflow_working_day(self,task_batch,working_day):
        """修改当前工作日 """
        
        update_sql = "update workflow_task set cur_working_day=%s where task_batch=%s"
        self.dbaccessor.do_inserto_withparams(update_sql,(working_day,task_batch))
        
    def get_task_working_day(self,task_batch):   
        """取得当前工作日 """
        
        rows = self.dbaccessor.do_query("select cur_working_day from workflow_task where task_batch={}".format(task_batch))
        if len(rows)==0:
            return None
        return rows[0][0]
        
    def update_workflow_current_task(self,task_batch,subtask_id):
        """修改工作流任务状态 """
        
        update_sql = "update workflow_task set cur_task_detail_id=%s where task_batch=%s"
        self.dbaccessor.do_inserto_withparams(update_sql,(subtask_id,task_batch))
               
    def get_workflow_config(self,workflow_id):
        """获取工作流配置 """

        row = self.dbaccessor.do_query("select id,type,name,work_dir,start_date,end_date from workflow where id={}".format(workflow_id))[0]
        rtn_obj = {"id":row[0],"type":row[1],"name":row[2],"work_dir":row[3],"start_date":row[4],"end_date":row[5]}
        return rtn_obj     
    
    def get_workflow_details_config(self,workflow_id):
        """获取子工作流配置 """
        
        sql = "select id,name,type,sequence,main_yaml_template,start_date,end_date," \
            "frequency,dynamic_config from workflow_detail where workflow_id={} and del_flag=0 order by sequence asc".format(workflow_id)
        rows = self.dbaccessor.do_query(sql)
        rtn_obj = [{"id":row[0],"name":row[1],"type":row[2],
                    "sequence":row[3],"main_yaml_template":row[4],
                    "start_date":row[5],"end_date":row[6],"frequency":row[7],"dynamic_config":row[8]} for row in rows]
        return rtn_obj    
    
    def create_subtask(self,main_task,config,working_day):
        """生成工作流数据 """
        
        # 累加本次主任务以及子任务类型对应的序号
        sequence = self.dbaccessor.do_query("select max(sequence) from workflow_task_detail where main_task_id=%s and workflow_detail_id=%s",
                                            (main_task["id"],config["id"]))[0][0]
        if sequence is None:
            sequence = 1
        else:
            sequence += 1
        insert_sql = "insert into workflow_task_detail(main_task_id,workflow_detail_id,sequence,status,working_day) values(%s,%s,%s,%s,%s)" 
        self.dbaccessor.do_inserto_withparams(insert_sql,(main_task["id"],config["id"],sequence,WorkflowSubStatus.created.value,working_day))
        return self.get_subtask(task_id=None, main_task_id=main_task["id"], config_id=config["id"], working_day=working_day)
        
    def get_subtask(self,task_id=None,main_task_id=None,config_id=None,working_day=None):
        """获取子工作任务数据
            Params:
                 task_id: 任务标识，如果不为空，则只使用此标识查询
                 main_task_id: 主任务标识
                 config_id： 子任务对应的子工作流计划标识
        """
        
        if task_id is not None:
            sql = "select id,main_task_id,workflow_detail_id,sequence,status,working_day from workflow_task_detail where id={}".format(task_id)
        else:
            sql = "select id,main_task_id,workflow_detail_id,sequence,status,working_day from workflow_task_detail" \
                " where main_task_id={} and workflow_detail_id={} and working_day={}".format(main_task_id,config_id,int(working_day))
        rows = self.dbaccessor.do_query(sql)
        if len(rows)==0:
            return None
        row = rows[0]
        rtn_obj = {"id":row[0],"main_task_id":row[1],"workflow_detail_id":row[2],"sequence":row[3],"status":row[4],"working_day":row[5]}
        return rtn_obj  

    def get_subtask_by_type_and_seq(self,main_task_id,config_id,sequence):
        """根据序号以及子任务类型获取子工作任务数据 
            Params:
                 main_task: 主任务标识
                 task_type: 子任务类别
                 sequence： 子任务序号
        """        
        
        sql = "select id,main_task_id,workflow_detail_id,sequence,status,working_day from workflow_task_detail" \
                " where main_task_id={} and workflow_detail_id={} and sequence={}".format(main_task_id,config_id,int(sequence))
        rows = self.dbaccessor.do_query(sql)
        if len(rows)==0:
            return None
        row = rows[0]
        rtn_obj = {"id":row[0],"main_task_id":row[1],"workflow_detail_id":row[2],"sequence":row[3],"status":row[4],"working_day":row[5]}
        return rtn_obj  
        
    def update_workflow_subtask_status(self,id,status,reset_sequence=False,init_sequence=False):
        """修改子工作流任务状态"""
        
        update_sql = "update workflow_task_detail set status=%s where id=%s"
        params = (status,id)
        # 根据标志，决定是否重置序号
        if reset_sequence:
            update_sql = "update workflow_task_detail set status=%s,sequence=0 where id=%s"
        if init_sequence:
            update_sql = "update workflow_task_detail set status=%s,sequence=1 where id=%s"            
        self.dbaccessor.do_inserto_withparams(update_sql,params)    

    def get_busi_task_id(self,subtask_id):    
        row = self.dbaccessor.do_query("select busi_task_id from workflow_task_detail where id={}".format(subtask_id))[0]
        return row[0]   
               
    def attach_busi_task_id(self,subtask_id,busi_task_id):    
        update_sql = "update workflow_task_detail set busi_task_id=%s where id=%s"
        self.dbaccessor.do_inserto_withparams(update_sql,(busi_task_id,subtask_id))  
              
    def clear_task_state(self,task_batch):
        """清空相关任务记录"""
        
        del_sql = "delete from workflow_task_detail where main_task_id in (select id from workflow_task where task_batch={})".format(task_batch)
        # 删除子表所有相关记录
        self.dbaccessor.do_updateto(del_sql)
        # 给主表相关字段置空
        upt_sql = "update workflow_task set cur_task_detail_id=0,cur_working_day=0 where task_batch={}".format(task_batch)
        self.dbaccessor.do_updateto(upt_sql)

    def get_pred_result_by_task_and_working_day(self,subtask_id,working_day):  
        """根据任务标号和工作日，取得之前生成的预测数据文件"""
        
        # 通过当前任务编号，找到对应的主任务编号。在根据工作日找到对应的序号，从而得到实际的预测任务编号，进一步取得对应的预测结果
        seq_sql = "select wc.sequence,wtd.main_task_id from workflow_calendar wc,workflow_task_detail wtd where wc.task_id=wtd.main_task_id " \
            "and wtd.id={} and wc.working_day={}".format(subtask_id,working_day)
        row = self.dbaccessor.do_query(seq_sql)[0]
        sequence = row[0]
        main_task_id = row[1]
        sql = "select pred_result_file from pred_result where task_id=(select wtd.id from workflow_task_detail wtd,workflow_detail wd,common_dict dict" \
                                       " where wtd.main_task_id={} and wtd.sequence={} and wtd.workflow_detail_id=wd.id and wd.type=dict.id" \
                                       " and dict.code='{}')".format(main_task_id,sequence,CommonDictEnum.WORK_TYPE__PRED.value)
        row = self.dbaccessor.do_query(sql)[0]
        return row[0]   
            
        
        