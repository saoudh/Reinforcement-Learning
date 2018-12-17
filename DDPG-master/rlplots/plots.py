import matplotlib.pyplot as plt
from rllab import getJsonDataFromConfigFile
from rllab import get_log_files,get_log_files_with_conf_interv_from_binary_file
import numpy as np
import sys
import re


class plot:
    def __init__(self,json_settings=None, log_dir=None,is_conf_interv=False):
        self.algs_names_1_layer = ["ddpg_without_bn_1_layer",
                        "mbddpg_without_bn_1_layer","ddpg_with_bn_1_layer","mbddpg_with_bn_1_layer"]
        self.colors = ["r", "b", "g", "y", "k", "c", "m"]
        self.colors_dict = [{"color": "r", "linestyle": None}, {"color": "b", "linestyle": None},
                            {"color": "g", "linestyle": None},
                            {"color": "y", "linestyle": None}, {"color": "k", "linestyle": None},
                            {"color": "c", "linestyle": None},
                            {"color": "m", "linestyle": None},
                            {"color": "r", "linestyle": "--"}, {"color": "b", "linestyle": "--"},
                            {"color": "g", "linestyle": "--"},
                            {"color": "y", "linestyle": "--"}, {"color": "k", "linestyle": "--"},
                            {"color": "c", "linestyle": "--"},
                            {"color": "m", "linestyle": "--"}
                            ]
        self.log_dir=log_dir
        # assign the right function for getting log-files, either with confidence interval or without
        self.get_log_files=get_log_files_with_conf_interv_from_binary_file if is_conf_interv else get_log_files
        # if only one Algorithm is plotted, then cast json_settings-object of that alg. to an array to process
        # correctly
        if not isinstance(json_settings,(list,)):
            self.json_settings=[json_settings]
        else:
            self.json_settings = json_settings
        self._preprocess()

    def _preprocess(self):
        # read data from files
        self.settings_arr=[]
        self.log_files=[]
        for i in self.json_settings:
            settings=i["settings"]
            #print("settings=",settings)
            settings=getJsonDataFromConfigFile(settings)
            self.settings_arr.append(settings)
            self.log_files.append(self.get_log_files(alg=settings["algorithm"]["type"],values=settings["plot"],log_subdir=self.log_dir))
        #print("logfiles.shape=",np.shape(self.log_files))
        #print("preprocess- finished-logfiles=",self.log_files)
        print("logfiles1.shape=", np.shape(self.log_files))


    def _preprocess_performance_neurons(self):
        # read data from files
        # all_logs_per_alg contains all alg like DDPG,DDPG-BN,MBDDPG as an array-element with
        # dict of information of form: {"actor_loss":array-data,"critic_loss":array-data ...}
        self.all_logs_per_alg = []
        for alg_name in self.algs_names_1_layer:
            mydict = {}
            for i,alg in enumerate(self.log_files):
                current_alg_name=self.json_settings[i]["alg"]
                settings=self.json_settings[i]["settings"]
                # print("settings=",settings)
                settings = getJsonDataFromConfigFile(settings)

                if current_alg_name.startswith(alg_name):
                    # add number of neurons to the dict
                    neurons = int(settings["actor_network"]["layers"]["layer0"])
                    if "neurons" in mydict:
                        mydict["neurons"]=np.hstack((mydict["neurons"],neurons))
                    else:
                        mydict={"neurons":np.array(neurons)}
                    mydict.update({"alg": alg_name})
                    mydict.update({"name_formatted":settings["algorithm"]["name_formatted"]})

                    for key in alg:
                        # if loss-type has already a value, then add new value to this type of loss
                        # only when array is not empty
                        if key in mydict and np.shape(alg[key])[0]>0:
                            # get only last value, after last episode
                            mydict[key]=np.vstack((mydict[key],alg[key][-1,:]))
                            print("alg[key][-1,:].shape=",np.shape(alg[key][-1,:]))
                            print("alg[key][-1,:].shape=",np.shape(alg[key]))

                            # and get mean value of the loss type and put it to a new dict-entry
                        elif np.shape(alg[key])[0]>0:
                            mydict.update({key:np.array(alg[key][-1,:])})
                    # calculate the mean of the reward, as the reward has because of confidence interval the shape of (-1,3)
                    # the mean has to be calculated across axis 0
                    if "reward_mean" in mydict:
                        mydict["reward_mean"] = np.vstack((mydict["reward_mean"], np.mean(alg["reward"],axis=0)))
                    else:
                        mydict.update({"reward_mean": np.array(np.mean(alg["reward"],axis=0))})
                    print("alg=",alg_name,"mydict[reward_mean]=",mydict["reward_mean"])
                    print("alg=",alg_name,"mydict[reward]=",mydict["reward"])


            if len(mydict)==0:
                continue
            self.all_logs_per_alg.append(mydict)
        for i in range(len(self.all_logs_per_alg)):
            print(i,"-alg=",self.all_logs_per_alg[i]["alg"],"-reward-mean=",self.all_logs_per_alg[i]["reward_mean"][:2])
            print(i,"-reward=",self.all_logs_per_alg[i]["reward"][:2])



    def plot_all_algs_as_performance_nr_of_neurons(self):
        #todo: instead of 0 the index with max. number of loss-type
        #N = len(self.settings_arr[0])
        # default value of maximum plots which is modified in this function
        N=3
        #fig = plt.clf()
        fig = plt.figure()
        fig.canvas.set_window_title(self.log_dir)
        myhandles=set()
        # max_nr_of_loss_types is the maximum len of loss-types per algorithm
        # i.e. DDPG has 5 loss-types like actor-loss,critic-loss, but MBDDPG has 7 loss types like model-error
        max=0
        idx_with_max_nr_of_loss_types = 0
        for k,i in enumerate(self.all_logs_per_alg):
            if len(i) >  max:
                # maximum number of plots depending on the algorithm which contains most loss-types
                N=len(i)
                max = len(i)
                idx_with_max_nr_of_loss_types=k
        #self.all_logs_per_alg[idx_with_max_nr_of_loss_types]=[k for i, k in enumerate(self.all_logs_per_alg[idx_with_max_nr_of_loss_types]) if k!="alg" and k!="neurons"]
        # because all_logs_per_alg array of dictionaries contains entries which doesn't have to be plotted like alg-name and neurons-number
        # we have to correct the index for plots
        difference=0
        # loop over all loss types names, i.e. k=actor_loss or critic_loss
        for i, k in enumerate(self.all_logs_per_alg[idx_with_max_nr_of_loss_types]):
            print("enumerate k=",k)

            # just process loss types in the array and skip alg-name and neurons-number
            if k=="alg" or k=="neurons" or k=="name_formatted":
                difference=difference+1
                continue
            plt.subplot(N,1,(i-difference)+1)
            plt.title(" ".join(k.split("_")))
            plt.xlabel("number of neurons")
            # settings_arr[0] contains the json-settings of the first algorithm, i.e. DDPG
            # log_files[0][k] contains File-Object, i.e. k="actor_error" the file actor_error.txt,
            # "critic_error" the file critic_error.txt etc.
            # j are the different algorithm, i.e. log_files[0]->DDPG with 25 Neurons, log_files[j=1]->DDPG with 50 neurons
            # loop over all algorithm like MBDDPG-25-neurons,DDPG-50-Neurons
            for j in range(len(self.all_logs_per_alg)):
                    # ddpg doesn't have model and reward NN
                    try:
                        # if array is empty, then skip it to avoid error output
                        print("k=",k)
                        print("j=",j)
                        print("self.all_logs_per_alg[j][k]=",self.all_logs_per_alg[j][k])
                        if np.shape(self.all_logs_per_alg[j][k])[0]>1:
                                #print(self.settings_arr[j]["algorithm"]["type"])
                                if  self.all_logs_per_alg[j]["alg"].startswith("ddpg_with_"):
                                    print("critic?alg=", self.all_logs_per_alg[j]["alg"], "-key=", k)
                                    print("self.all_logs_per_alg[j][k].shape=",np.shape(self.all_logs_per_alg[j][k]))

                                try:
                                    print("number of y=",len(np.stack(self.all_logs_per_alg[j][k],1)[0]))
                                    print("self.all_logs_per_alg[j][neurons]",self.all_logs_per_alg[j]["neurons"])
                                    handle=self._plot_mean_and_confid_interv(np.stack(self.all_logs_per_alg[j][k],1)[0],
                                         np.stack(self.all_logs_per_alg[j][k],1)[1], np.stack(self.all_logs_per_alg[j][k],1)[2],
                                         color_mean=self.colors_dict[j%len(self.colors_dict)]["color"], color_shading=self.colors_dict[j%len(self.colors_dict)]["color"],
                                              linestyle=self.colors_dict[j%len(self.colors_dict)]["linestyle"],label=self.all_logs_per_alg[j]["name_formatted"],x_values=self.all_logs_per_alg[j]["neurons"])
                                except IndexError:
                                    print("Indexerror-j=",j,"-k=",k)
                                    print("self.all_logs_per_alg[j][k])=",self.all_logs_per_alg[j][k])
                        # to add only one handle per algorithm and not for every error-type of every algorithm
                        # because we then get multiple duplicates of every algorithm as they have several error-logs
                        #if k == "actor_error":
                        #todo: it plots only the reward, see json file in "plots"
                        if k == "reward":

                                # add handle for the plot-legend
                            #todo: it is need to sort the algorithm according to the algorithm-type and nr. of neurons
                            myhandles.add(handle)
                    except KeyError:
                        print("Keyerror-j=",j,"-k=",k)
                        continue
        # with mbddpg there was an error with the legend-display ony the plot, only ddpg worked
        # thats why it is necessary to add the handles manually
        plt.legend(handles=myhandles)
        #plt.legend(loc="upper right")
        plt.show()
        plt.pause(0.0000001)


    # only one loss type compared between algorithms and with/without BN
    def plot_all_algs_as_performance_nr_of_neurons_per_loss_type(self, loss_type):
        fig = plt.figure(figsize=(7, 3))
        fig.canvas.set_window_title(self.log_dir)
        myhandles=set()
        # k is the loss type to be plotted
        k=loss_type
        color_idx=color_idx_with_bn=color_idx_without_bn=0
        myhandles = dict()

        for j in range(len(self.all_logs_per_alg)):
            for k in loss_type:
                # settings_arr[0] contains the json-settings of the first algorithm, i.e. DDPG
                    # log_files[0][k] contains File-Object, i.e. k="actor_error" the file actor_error.txt,
                    # "critic_error" the file critic_error.txt etc.
                    # j are the different algorithm, i.e. log_files[0]->DDPG with 25 Neurons, log_files[j=1]->DDPG with 50 neurons
                    # loop over all algorithm like MBDDPG-25-neurons,DDPG-50-Neurons

                    # 2 rows for Batch normalization and without BN data
                    # BN plots are in the first row and without BN in the second row
                    x=1 if "with_bn" in self.all_logs_per_alg[j]["alg"] else 2
                    if x==1:
                        color_idx=color_idx_with_bn
                        color_idx_with_bn+=1
                    elif x==2:
                        color_idx=color_idx_without_bn
                        color_idx_without_bn+=1
                    plt.subplot(2,1,x)
                    title="with Batch Normalization" if x==1 else "without Batch Normalization"
                    plt.title(title)
                    plt.xlabel("number of neurons")
                    plt.ylabel("$J(\pi)$")
                    handle=0
                    # ddpg doesn't have model and reward NN
                    try:
                        # if array is empty, then skip it to avoid error output
                        if np.shape(self.all_logs_per_alg[j][k])[0]>1:
                                if self.all_logs_per_alg[j]["name_formatted"].split(" ")[0]=="DDPG":
                                    print("x=",x,"-",self.all_logs_per_alg[j]["name_formatted"].split(" ")[0],k,"=",self.all_logs_per_alg[j][k][:5])
                                try:
                                    alg_name=self.all_logs_per_alg[j]["name_formatted"].split(" ")[0]
                                    label=alg_name+" $J(\pi_N)$" if k=="reward" else alg_name+" $N^{-1}\sum_i J(\pi_i)$"
                                    handle=self._plot_mean_and_confid_interv(np.stack(self.all_logs_per_alg[j][k],1)[0],
                                         np.stack(self.all_logs_per_alg[j][k],1)[1], np.stack(self.all_logs_per_alg[j][k],1)[2],
                                         color_mean=self.colors_dict[color_idx%len(self.colors_dict)]["color"], color_shading=self.colors_dict[color_idx%len(self.colors_dict)]["color"],
                                              linestyle=self.colors_dict[color_idx%len(self.colors_dict)]["linestyle"],label=label,x_values=self.all_logs_per_alg[j]["neurons"])
                                except IndexError:
                                    print("Indexerror-j=",j,"-k=",k)
                                    print("self.all_logs_per_alg[j][k])=",self.all_logs_per_alg[j][k])

                        # every plot row has its own handles/legend e.g. Without BN and with BN
                        myhandles[x].append(handle) if x in myhandles else myhandles.update({x:[handle]})

                    except KeyError:
                        print("Keyerror-j=",j,"-k=",k)
                        continue
                    plt.legend(handles=myhandles[x])

        # top is the space of the top of the figure to the plot->the higher the less space to the top
        # hspace is the horizontal space between the plots
        plt.subplots_adjust(top=0.9, bottom=0.2, hspace=0.8)
        plt.show()
        plt.pause(0.0000001)



    def _plot_mean_and_confid_interv(self,mean, lb, ub, color_mean=None, color_shading=None,linestyle=None,label=None,x_values=None):
        # plot the shaded range of the confidence intervals
        if x_values is None:
            plt.fill_between(range(mean.shape[0]), ub, lb,
                         color=color_shading, alpha=.5)
            h,=plt.plot(mean, color=color_mean,linestyle=linestyle,label=label)

        else:
            plt.fill_between(x_values, ub, lb,
                             color=color_shading, alpha=.5)
            h,=plt.plot(x_values,mean, color=color_mean,linestyle=linestyle,label=label)

        # plot the mean on top
        return h

    def plot_all_algs_conf_interv(self):
        N = len(self.settings_arr[0])
        #fig = plt.clf()
        fig = plt.figure()
        fig.canvas.set_window_title(self.log_dir)
        myhandles=set()
        # max_nr_of_loss_types is the maximum len of loss-types per algorithm
        # i.e. DDPG has 5 loss-types like actor-loss,critic-loss, but MBDDPG has 7 loss types like model-error
        #idx_with_max_nr_of_loss_types=np.max([len(i) for i in self.log_files])
        max = 0
        idx_with_max_nr_of_loss_types = 0
        for k, i in enumerate(self.log_files):
            if len(i) > max:
                max = len(i)
                idx_with_max_nr_of_loss_types = k
        # loop over all loss types names, i.e. k=actor_loss or critic_loss
        for i, k in enumerate(self.log_files[idx_with_max_nr_of_loss_types]):
            plt.subplot(N,1,i+1)
            # format the plot name by replacing underscore with empty whitespace, e.g. "actor error" instead of "actor_error"
            plt.title(" ".join(k.split("_")))
            plt.xlabel("episode")
            # settings_arr[0] contains the json-settings of the first algorithm, i.e. DDPG
            # log_files[0][k] contains File-Object, i.e. k="actor_error" the file actor_error.txt,
            # "critic_error" the file critic_error.txt etc.
            # j are the different algorithm, i.e. log_files[0]->DDPG with 25 Neurons, log_files[j=1]->DDPG with 50 neurons
            # loop over all algorithm like MBDDPG-25-neurons,DDPG-50-Neurons
            for j in range(len(self.log_files)):
                    # ddpg doesn't have model and reward NN

                    try:
                        # if array is empty, then skip it to avoid error output
                        if np.shape(self.log_files[j][k])[0]>1:
                                #print(self.settings_arr[j]["algorithm"]["type"])
                                try:
                                    print("self.logfiles[j][k]=", self.log_files[j][k])
                                    print("np.stack(self.logfiles[j][k],1)=",
                                          np.stack(self.log_files[j][k], 1))
                                    print("np.stack(self.logfiles[j][k],1)[0]=",
                                          np.stack(self.log_files[j][k], 1)[0])

                                    handle=self._plot_mean_and_confid_interv(np.stack(self.log_files[j][k],1)[0],
                                         np.stack(self.log_files[j][k],1)[1], np.stack(self.log_files[j][k],1)[2],
                                         color_mean=self.colors_dict[j%len(self.colors_dict)]["color"], color_shading=self.colors_dict[j%len(self.colors_dict)]["color"],
                                              linestyle=self.colors_dict[j%len(self.colors_dict)]["linestyle"],label=self.settings_arr[j]["algorithm"]["name_formatted"])
                                except IndexError:
                                    print("Indexerror-j=",j,"-k=",k)
                        # to add only one handle per algorithm and not for every error-type of every algorithm
                        # because we then get multiple duplicates of every algorithm as they have several error-logs
                        if k == "actor_error":
                            # add handle for the plot-legend
                            #todo: it is need to sort the algorithm according to the algorithm-type and nr. of neurons
                            myhandles.add(handle)
                    except KeyError:
                        print("Keyerror-j=",j,"-k=",k)
                        continue
        # with mbddpg there was an error with the legend-display ony the plot, only ddpg worked
        # thats why it is necessary to add the handles manually
        plt.legend(handles=myhandles,loc="best")
        plt.show()
        plt.pause(0.0000001)

    def plot_all_algs(self):
        N = len(self.settings_arr[0])
        #plt.clf()
        fig = plt.figure()
        fig.canvas.set_window_title(self.log_dir)
        print(self.log_dir)
        colors = ["r", "b", "g"]
        for i, k in enumerate(self.log_files[len(self.log_files)-1]):

            plt.subplot(N,1,i+1)
            plt.title(k)

            # settings_arr[0] contains the json-settings of the first algorithm, i.e. DDPG
            # log_files[0][k] contains File-Object, i.e. k="actor_error" the file actor_error.txt,
            # "critic_error" the file critic_error.txt etc.
            for j in range(len(self.log_files)):
                # ddpg doesn't have model and reward NN
                if k in self.log_files[j]:
                    #if j==0 and k=="critic_error":
                        plt.plot(self.log_files[j][k], colors[j%len(self.log_files)],label=self.settings_arr[j]["algorithm"]["type"])
        plt.legend(loc="upper right")
        plt.show()
        plt.pause(0.0000001)



algs_1_layer=["ddpg_with_bn_1_layer","ddpg_without_bn_1_layer","mbddpg_with_bn_1_layer","mbddpg_without_bn_1_layer"]
#algs_1_layer=["mbddpg_with_bn_1_layer","mbddpg_without_bn_1_layer"]#,"ddpg_without_bn_1_layer"]

jobs_1_layer=[]
nr_neurons=np.array([25,50,75,100,125,150,175,200])
#nr_neurons=np.array([25])
myarr=[]
for i in algs_1_layer:
    for j in nr_neurons:
        myarr.append({"settings": i + "_" + str(j) + "_neurons.json", "alg": i + "_" + str(j) + "_neurons"})

jobs_1_layer=myarr

# sub-directory name within logfiles-directory which has as the name the parameter specification
# i.e. subdirectory name "pendelum_100_neurons_20_episodes"
logdirs=[]
#logdirs.append("qube")

logdirs.append("pendelum")

# whether to use confidence interval or not
plot_without_conf_interv=False
plot_with_conf_interv=False
is_conf_interv=True
plot_performance_per_neurons=False
plot_performance_per_neurons_per_loss_type=True




def run_plot(is_conf_interv):
    # for k,i in enumerate(logdirs):
    myplot = plot(jobs_1_layer, log_dir=logdirs[0], is_conf_interv=is_conf_interv)
    if plot_with_conf_interv:
        myplot.plot_all_algs_conf_interv()
    elif plot_performance_per_neurons:
        myplot._preprocess_performance_neurons()
        myplot.plot_all_algs_as_performance_nr_of_neurons()
    elif plot_performance_per_neurons_per_loss_type:
        myplot._preprocess_performance_neurons()
        # plot only the loss-type/performance put in the argument
        myplot.plot_all_algs_as_performance_nr_of_neurons_per_loss_type(["reward","reward_mean"])
    elif plot_without_conf_interv:
        myplot.plot_all_algs()

run_plot(is_conf_interv)