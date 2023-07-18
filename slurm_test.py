import sys

print("task %f is done" %sys.argv[1])

txt_file = open("test.txt", "a")
txt_file.write("%f" %sys.argv[1])
txt_file.close()
