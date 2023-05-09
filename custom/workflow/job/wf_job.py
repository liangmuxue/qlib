import schedule
import time
from datetime import datetime
from workflow.workflow_task import WorkflowTask

def job():
    print("I'm working...")
    task = WorkflowTask(task_batch=118,workflow_name="wf_backtest_flow_2023",resume=True)  
    task.start_task()
    

schedule.every().day.at("00:30").do(job)
while True:
    schedule.run_pending()
    print("time is:{}".format(datetime.now()))
    time.sleep(3)