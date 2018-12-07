# -*- coding: utf-8 -*-
import socket
import sys

if __name__ == '__main__':
	host = 'localhost'
	port = 8888
	try:
		fi = "./input/testMail.txt"
		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		msg = open(fi, encoding='utf-8').read()
		s.connect((host, port))
		s.sendall(msg.encode())
		s.close()
		print("发送成功！")
	except:
		print("error: Input the email location")

