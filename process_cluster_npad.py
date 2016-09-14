# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 17:53:14 2015

@author: rennocosta
"""


import subprocess 
import argparse


#process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
#process.wait()


from multiprocessing import Lock, Process, Queue, current_process

def worker(work_queue,work_lock,fnameout):
    try:
        isempty = False
        while not isempty:
            work_lock.acquire()
            isempty = work_queue.empty()
            if (not isempty):        
                command = work_queue.get()
                print(('aaa: %s') % (command))
            work_lock.release()
            if (not isempty):   
                #command = ['/run/myscript', '--arg', 'value']
                #subprocess.Popen(command).wait()
                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
                process.wait()
                work_lock.acquire()
                try:
                    with open(fnameout, "a") as myfile:
                        myfile.write(command + "\n")
                except:
                    pass
                work_lock.release()
    except: 
        pass
    return True

def main():
    
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('run_num', metavar='run_num', type=int, nargs=1,
                   help='run_num')    
    parser.add_argument('workers', metavar='workers', type=int, nargs=1,
                   help='workers')   
    
    args = parser.parse_args()
    run_num = args.run_num[0]
    # read in the 

    fname = "npad_input_file_" + str(run_num) + ".txt"
    fnameout = "npad_output_file_" + str(run_num) + ".txt"

    with open(fname) as fff:
        commands = fff.readlines()

    #commands = (
    #"python3 stable_cluster_singlenet_learned.py 89 89 3 1 1 0 100 10 12 60 15 80 > XUXA_011.txt",
    #"python3 stable_cluster_singlenet_learned.py 89 89 3 1 1 0 100 10 13 60 15 80 > XUXA_012.txt"
    #)

    workers = args.workers[0]
    work_queue = Queue()
    work_lock = Lock()

    processes = []

    for command in commands:
        work_queue.put(command)

    for www in range(workers):
        ppp = Process(target=worker, args=(work_queue,work_lock,fnameout))
        ppp.start()
        processes.append(ppp)
     
    for ppp in processes:
        ppp.join()


            
            
    
    
if __name__ == '__main__':
    main()