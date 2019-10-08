# -- coding:utf-8 --
import hashlib
import requests
import json
import time

"""
@FileName : JLU选课脚本
@Author : 圣西罗卡卡
@QQ : 435390736
@Create date : 20191008
@description : 用于JLU选课，需要提前将要选择的课程添加至快捷选课界面，在选课开始前几分钟启动此脚本。
"""


class Select_Course:

    def __init__(self, user_id, pwd):
        self.user_id = user_id
        self.pwd = pwd
        self.lesson_id_list = []
        self.session = requests.Session()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36",
            "Origin": "https://uims.jlu.edu.cn",
            "Host": "uims.jlu.edu.cn"
        }


    def Login(self):
        self.pwd = "UIMS" + self.user_id + self.pwd
        m = hashlib.md5()
        m.update(bytes(self.pwd, encoding="utf-8"))
        self.pwd = m.hexdigest()
        login_url = "https://uims.jlu.edu.cn/ntms/j_spring_security_check"
        self.headers["Referer"] = "https://uims.jlu.edu.cn/ntms/userLogin.jsp?reason=loginError"
        data = {
            "j_username": self.user_id,
            "j_password": self.pwd,
            "mousePath": "THQABTAQDqTAwDwTBQD3TBwD/SCgEFSDQENREQETQFQEaQGQEhQHQEoQIQEvPJAE2PKAE9PLAFDOMQFKMNgFRMPAFZLQQFfLRwFnKSwFtKTwF0JUQF8JVAGCIVQGIIWAGPIWQGWIWgGdIWwGkIXQGrIXQGyIXgG6IXwHAIYAHNIYQHUIYQHiIYwHpHZAHwHZgH3HaQH+GawIFGbQIMGcQITGdAIZGdgIiGeQInGfAIuGfgI1GgAJDHgAJRHgQJXIgQJzIgQKBJgAKjJfwKqJfgLGJfQLUIfQMEIfwMfIgAMnIhAMtIiwM0IlAM7IoANCIrQNJIuQNQIxANYIzgNiI2QNlI4QNsI6QNzKHgN1",
        }
        self.session.headers = self.headers
        res = self.session.post(login_url, data=data)


    def Get_Courses(self):
        service_url = "https://uims.jlu.edu.cn/ntms/service/res.do"
        course_data = {
            "tag": "lessonSelectLogTcm@selectGlobalStore",
            "branch": "quick",
            "params": {
                'splanId': '970',
            }
        }
        self.headers["Referer"] = "https://uims.jlu.edu.cn/ntms/index.do"
        self.headers['Content-Type'] = 'application/json'
        response = self.session.post(service_url, headers=self.headers, data=json.dumps(course_data))
        msg = response.content.decode("utf-8")
        for i in range(len(json.loads(msg)["value"])):
            lesson_id = json.loads(msg)["value"][i]["lsltId"]
            self.lesson_id_list.append(lesson_id)


    def Select_Course(self, lesson_id):
        service_url = "https://uims.jlu.edu.cn/ntms/action/select/select-lesson.do"
        post_lesson_data = {
            "lsltId": lesson_id,
            "opType": "Y"
        }
        response = self.session.post(service_url, headers=self.headers, data=json.dumps(post_lesson_data))
        back_datas = json.loads(response.content.decode("utf-8"))
        if not back_datas["msg"]:
            print("成功选课一门!")
            return lesson_id
        else:
            print(back_datas["msg"])
            return 0



if __name__ == "__main__":
    user_id = input("请输入账号:")
    pwd = input("请输入密码:")
    program = Select_Course(user_id, pwd)
    program.Login()
    program.Get_Courses()

    tic = time.time()
    while( 1 ):
        try:
            for id in program.lesson_id_list:
                flag = program.Select_Course(id)
                if flag:program.lesson_id_list.remove(flag)

                toc = time.time()
                if toc - tic > 180:
                    tic = time.time()
                    program = Select_Course(user_id, pwd)
                    program.Login()
                    program.Get_Courses()
                    print("*"*30, "\t已重新登陆\t", "*"*30)
        except Exception:pass