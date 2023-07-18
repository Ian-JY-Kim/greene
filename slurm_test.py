import sys

print("task %d is done" %int(sys.argv[1]))

txt_file = open("test.txt", "a")
txt_file.write("%d" %int(sys.argv[1]))
txt_file.close()
