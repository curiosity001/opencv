import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QMessageBox

def on_login_button_click():
    entered_username = username_input.text()
    entered_password = password_input.text()

    if entered_username == predefined_username and entered_password == predefined_password:
        # 登录成功
        QMessageBox.information(window, "登录成功", "登录成功！")
        # 在这里添加你的上位机界面代码

    else:
        # 登录失败
        QMessageBox.warning(window, "登录失败", "用户名或密码错误！")

def on_register_button_click():
    entered_username = username_input.text()
    entered_password = password_input.text()

    # 在这里添加你的注册逻辑，例如将账号和密码保存到数据库或文件中

    QMessageBox.information(window, "注册成功", "注册成功！")

# 创建应用程序对象
app = QApplication(sys.argv)

# 创建窗口对象
window = QWidget()
window.setWindowTitle("上位机界面")

# 创建布局
layout = QVBoxLayout(window)

# 创建用户名标签和文本框
username_label = QLabel("用户名:")
layout.addWidget(username_label)
username_input = QLineEdit()
layout.addWidget(username_input)

# 创建密码标签和文本框
password_label = QLabel("密码:")
layout.addWidget(password_label)
password_input = QLineEdit()
password_input.setEchoMode(QLineEdit.Password)
layout.addWidget(password_input)

# 创建登录按钮
login_button = QPushButton("登录")
login_button.clicked.connect(on_login_button_click)
layout.addWidget(login_button)

# 创建注册按钮
register_button = QPushButton("注册")
register_button.clicked.connect(on_register_button_click)
layout.addWidget(register_button)

# 设置窗口大小
window.resize(300, 150)

# 显示窗口
window.show()

# 运行应用程序的主循环
sys.exit(app.exec_())
