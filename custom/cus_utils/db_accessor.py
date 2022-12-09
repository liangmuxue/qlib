import pymysql

class DbAccessor():

    def __init__(self, cfg):
        if cfg !={}:
            self.host= cfg["host"]
            self.port= cfg["port"]
            self.user= cfg["user"]
            self.password= cfg["password"]
            self.database= cfg["database"]
            return
            
        self.host="127.0.0.1"
        self.port = 3306
        self.user = "lmx"
        self.password = "lmx"
        self.database = "qlib"


    def get_connection(self):
        return pymysql.connect(host=self.host,
                               port=self.port,
                               user=self.user,
                               password=self.password,
                               database=self.database,
                               charset='utf8')

    def do_query(self,sql_str,params=None):
        con = self.get_connection()
        cur = con.cursor()
        if params is None:
            cur.execute(sql_str)
        else:
            cur.execute(sql_str,params)
        rows = cur.fetchall()
        cur.close()
        con.close()
        return rows

    def do_inserto(self, sql_str):
        con = self.get_connection()
        cur = con.cursor()
        try:
            cur.execute(sql_str)
            con.commit()
        except:
            con.rollback()
            print('Insert operation error')
            raise
        finally:
            cur.close()
            con.close()

    def do_updateto(self,sql_str):
        con = self.get_connection()
        cur = con.cursor()
        try:
            cur.execute(sql_str)
            con.commit()
        except:
            con.rollback()
            print('Insert operation error')
            raise
        finally:
            cur.close()
            con.close()

    def do_inserto_withparams(self,sql_str,params):
        con = self.get_connection()
        cur = con.cursor()
        try:
            cur.execute(sql_str,params)
            con.commit()
        except:
            con.rollback()
            print('Insert operation error')
            raise
        finally:
            cur.close()
            con.close()

    def do_update_withparams(self,sql_str,params):
        con = self.get_connection()
        cur = con.cursor()
        try:
            cur.execute(sql_str,params)
            con.commit()
        except:
            con.rollback()
            print('Insert operation error')
            raise
        finally:
            cur.close()
            con.close()