% function [R Rss a ut]=AfdStartOldRegress4(FileName,ar,un,tlag,tlead)
%% info
% this collects data back and ahead of time for observational task
% applies regression on force
%% info
clear all
close all
% clc
ar=3; %are of the brain 3 for s1
% un=32; %number of units to be used
tlag=500;%1000; % time in ms that you want to include as time lag
tlead=500;
fornot=1;
%% Variable keeps
kks=30; % smoothing SD for Spike
kkf=100; % smoothing SD for force
exsmooth=1; % 1 for smoothing of force data
task_type=0; % 0 for observational 1 for manual
derivative=0;
%% variable of parameters 
timebend=100;
tdat=0.90; %percent of data for training
tlag=floor(tlag/timebend);
tlead=floor(tlead/timebend);
%% Scene Codes:
% reset_scene: 0
% disp_rp_scene: 1
% reaching_scene: 2
% grasping_scene: 3
% transport_scene: 4
% releasing_scene: 5
% success_scene: 6
% penalty_scene: 7
% shattering_penalty_scene: 8close all
%% loading and initiation
if ~exist('FileName')
    [FileName,PathName] = uigetfile('*.mat','Select the MATLAB code file','MultiSelect', 'on');
    %ar=3;
else
    fornot=0;
end
if exist('Flag')
    Flag.state=4; 
    % which area of brain will be used
    Flag.area=ar;
    Flag.prevt=tlag/1000;  % time in second before center state
    Flag.postt=tlead/1000; %3.5;  %time in second after the center state
%    Flag.timebend=1; % width of the time bin in miliseconds
end
Flag.area=ar;
load(FileName);
%% analysis part - Parameters
%state= center state
state=Flag.state; 
% which area of brain will be used
area=Flag.area;
prevt=Flag.prevt;  % time in second before center state
postt=Flag.postt; %3.5;  %time in second after the center state
timebend=1; % width of the time bin in miliseconds

%% force smoothing
% task_data.force_sensor.force_gf=smoother(task_data.force_sensor.force_gf,500,20);

%% organize PMv data 
% this section is to avoid PMv data each area has pmv data after
% channe 96. so collect data before channel 96
% ut=AfdStartOldRegress2_Per_neuron(FileName,ar);
for i=1:3
    %clear tmp
    k=0;
    for j=1:length(neural_data.spikeTimes{1,i})
        if(neural_data.spikeTimes{1,i}{1,j}.ch<97)
            k=k+1;
            tmp{1,k}=neural_data.spikeTimes{1,i}{1,j};
        end
    end
    neural_data.spikeTimes{1,i}=[];
    neural_data.spikeTimes{1,i}=tmp;
    clear tmp
end
%% detect reward Level
a=find(task_data.strobe.nums==6);
% a(length(a))=[];
for i=1:length(a)    
    if(length(task_data.strobe.receipt_ts)>=(a(i)+1))
        b(i)=task_data.strobe.receipt_ts(a(i)+1)-task_data.strobe.receipt_ts(a(i));
    end
end
% plot(b)
count=[0 0 0 0];
for i=1:length(b)
    if(b(i)>2)
        c(i)=3;
        count(c(i)+1)=count(c(i)+1)+1;
    elseif(b(i)>1)
        c(i)=2;
        count(c(i)+1)=count(c(i)+1)+1;
    elseif(b(i)>0.5)
        c(i)=1;
        count(c(i)+1)=count(c(i)+1)+1;
    else
        c(i)=0;
        count(c(i)+1)=count(c(i)+1)+1;
    end
end

for i=1:length(c)
%     task_data.reward_num.val(i)=c(i);
%     task_data.reward_num.ts(i)=task_data.strobe.receipt_ts(a(i)-4);
%     task_data.reward_num.state(i)=task_data.strobe.nums(a(i)-4);
    reward_num.val(i)=c(i);
    reward_num.ts(i)=task_data.strobe.receipt_ts(a(i)-4);
    reward_num.state(i)=task_data.strobe.nums(a(i)-4);
end

%% Deviding the EMG data into trials from the time of a zero on the second column of 'stb' to next zero
%field variables for structure
field1 = 'trialId';  value1 = zeros(1);
field2 = 'spikes';  value2 = zeros(1);
%creating structure for saving data
data = struct(field1,value1,field2,value2);
%Marking the position of the states
t0=find(neural_data.Strobed(:,2)==0);
t6=find(neural_data.Strobed(:,2)==6);
t7=find(neural_data.Strobed(:,2)==7);

k=1;
l=1;
for i=1:length(t0)-1
    if((neural_data.Strobed(t0(i)+6,2)==6))%&&(neural_data.Strobed(t0(i)+7,2)==8)) %no reward for 7
        for j=0:6
            ttr(k,j+1)=neural_data.Strobed(t0(i)+j,1);
            tr(k,j+1)=neural_data.Strobed(t0(i)+j,2);
        end
       % ttr(k,j+2)=neural_data.Strobed(t0(i)+j,3);
        k=k+1;
    elseif((neural_data.Strobed(t0(i)+6,2)==7)||(neural_data.Strobed(t0(i)+5,2)==7)) %reward for 6
        for j=0:5
           % if(
                ttn(l,j+1)=neural_data.Strobed(t0(i)+j,1);
                tn(l,j+1)=neural_data.Strobed(t0(i)+j,2);
        end
       % ttn(l,j+2)=neural_data.Strobed(t0(i)+j,3);
        l=l+1;
    else
    end
end
% %check difference
% plot(ttr(:,2)-ttr(:,1))
% hold on
% plot(ttr(:,3)-ttr(:,1));
% hold off
%% Average Force sensing distance
for i=1:length(task_data.force_sensor.receipt_ts)-1
    tmp(i)=task_data.force_sensor.receipt_ts(i+1)-task_data.force_sensor.receipt_ts(i);
end
mdis=ceil(500/(mean(tmp)*1000));
tmp=[];
%% add reward nad Punishment level information for Rewarded trials (Trial marker matrix)
k=1;
l=0;
for i=1:(size(ttr,1))
    while(ttr(i,2)>reward_num.ts(k))
        k=k+1;
    end
    if((reward_num.ts(k)>ttr(i,2)) && (reward_num.ts(k)<ttr(i,4)))
        ttr(i,8)=reward_num.val(k);
        ttr(i,9)=reward_num.ts(k);
        k=k+1;
    else
        ttr(i,8)=0;
        ttr(i,9)=0;
    end
    
end
%% collecting all force vaule of non-zero tt=position ttm= time
k=0;
j=1;

for i=1:length(task_data.force_sensor.receipt_ts)
    if(task_data.force_sensor.force_gf(i)<0.000001)
        if(k>0)
            tt(j,2)=i-1;
            ttm(j,2)=task_data.force_sensor.receipt_ts(i-1);
            j=j+1;
            k=0;
        end
    elseif(i==length(task_data.force_sensor.receipt_ts))
        if(k>0)
            tt(j,2)=i-1;
            ttm(j,2)=task_data.force_sensor.receipt_ts(i);
            j=j+1;
            k=0;
        end
    else
        if(k==0)
            tt(j,1)=i;
            ttm(j,1)=task_data.force_sensor.receipt_ts(i);
            k=1;
        end
    end
end

%% removing the force values that doesn't have significant length
for i=1:size(tt,1)
    l(i)=(task_data.force_sensor.receipt_ts(tt(i,2))-task_data.force_sensor.receipt_ts(tt(i,1)));
    if((task_data.force_sensor.receipt_ts(tt(i,2))-task_data.force_sensor.receipt_ts(tt(i,1)))>0.3)
        k(i)=1;
    else
        k(i)=0;
    end
end
i=size(tt,1);
while(i>0)
    if(k(i)==0)
        tt(i,:)=[];
    end
    i=i-1;
end
%% adding extra values on force trial marker
if(exsmooth>0)
        % extra added
    i=1;
    tt(i,2)=tt(i,2)+mdis;
    if((tt(i,1)-mdis)<1)
        tt(i,1)=1;
    else
        tt(i,1)=tt(i,1)-mdis;
    end
    % extra added
    for i=2:size(tt,1)-1
        tt(i,1)=tt(i,1)-mdis;
        tt(i,2)=tt(i,2)+mdis;
    end
    i=i+1;
    tt(i,1)=tt(i,1)-mdis;
    if((tt(i,2)+mdis)>length(task_data.force_sensor.receipt_ts))
        tt(i,2)=length(task_data.force_sensor.receipt_ts);
    else
        tt(i,2)=tt(i,2)+mdis;
    end
end
%% creating force time matrix
clear ttm;
for i=1:size(tt,1)
    ttm(i,1)=task_data.force_sensor.receipt_ts(tt(i,1));
    ttm(i,2)=task_data.force_sensor.receipt_ts(tt(i,2));
end
%% adding force information time on the main trial marker matrix
for i=1:size(ttr,1)
    j=1;
    k=1;
    while((k>0)&&(j<=size(ttm,1)))
        if((ttr(i,5)>ttm(j,1)) && (ttr(i,5)<ttm(j,2)))%((ttm(j,2)>ttr(i,1)) && (ttm(j,2)<ttr(i,7)))|| ((ttm(j,1)>ttr(i,1)) && (ttm(j,1)<ttr(i,7))))
            ttr(i,10)=ttm(j,1);
            ttr(i,11)=ttm(j,2);
            ttr(i,12)=tt(j,1);
            ttr(i,13)=tt(j,2);
            k=0;
        end
        j=j+1;
    end
end
            
%% for rewarding data
for k=1:size(ttr,1) %k defines the number of trial
    if(ttr(k,10)>0)
        prevt=ttr(k,10)-tlag*100/1000;
        postt=ttr(k,11)+tlead*100/1000;
        a=round(1000*(prevt));
        b=round(1000*(postt));
%         a=round(1000*(ttr(k,state)-prevt)); %starting value for the spike rate counting
% %         postt=ttr(k,state+1)-ttr(k,state);
%         b=round(1000*(ttr(k,state)+postt)); %ending value for the spike rate counting
        data(k).trialId=k;%k;  % number of the trial
        data(k).Rnum=ttr(k,8);
        data(k).Forcetime=postt-prevt;
%         data(k).Pnum=ttr(k,10);
        data(k).spikes=zeros(size(neural_data.spikeTimes{1,area},2),round(1000*(postt-prevt)/timebend));

        for i=1:size(neural_data.spikeTimes{1,area},2)%arealength
            clear tempSpike %full spike time for the unit
            clear tempSpike2 %required spike time for the unit
                    tempSpike=round(1000*neural_data.spikeTimes{1,area}{1,i}.ts);
            x=1;
            for j=1:length(tempSpike)
                if(tempSpike(j)>a && tempSpike(j)<b)
                    tempSpike2(x)=tempSpike(j); 
                    x=x+1;
                end
            end
            %j=1;
            if(x>1)
                tempSpike2=floor((tempSpike2-a)/timebend);%+1; % removed this 1 need to check this part
                for j=1:length(tempSpike2)
                    data(k).spikes(i,tempSpike2(j))=data(k).spikes(i,tempSpike2(j))+1;
                end
            end
        end  
        j=0;
        for i=ttr(k,12):ttr(k,13)
            j=j+1;
            data(k).force(j)=task_data.force_sensor.force_gf(i);
            data(k).forceT(j)=task_data.force_sensor.receipt_ts(i);
            data(k).ftime(1)=ttr(k,10);
            data(k).ftime(2)=ttr(k,11);
        end
    end
end


%% Calculates the frequency of force
timebend=100;
tmp=[];
for i=1:length(task_data.force_sensor.receipt_ts)-1
    tmp(i)=task_data.force_sensor.receipt_ts(i+1)-task_data.force_sensor.receipt_ts(i);
end
mdis=ceil((mean(tmp)*1000));
tmp=[];
ft=ceil(timebend/mdis);
% ft=20
% if(task_type==1)
%     ft=timebend/5;
% else
%     ft=timebend/20;
% end

for i=1:size(data,2)
    tmp=[];
    for j=1:floor((size(data(i).spikes,2)-(tlag+tlead)*timebend)/timebend)%floor(size(data(i).force,2)/ft)%length(data(i).force)/ft
        tmp(j)=mean(data(i).force((j-1)*ft+1:j*ft));
    end
    data(i).force=[];
    data(i).force=tmp;
end

%% show plot 
for i=2:7
    avrg_state(i)=mean((tr(:,i)-tr(:,1))*1000)/timebend;
end
one=0;
for i=1:size(data,2)
    if(data(i).trialId>0)
        one=one+1;
    end
end

A=[one size(data,2)-one];
%dat=data;
%% sorting data

p='R';%'R/P'
for i=1:A(1)%size(data,2)
    if(p=='R')
        temp(i)=data(i).Rnum;%Rnum;
    else
        temp(i)=data(i).Pnum;%Rnum;
    end
end
for j=0:3
    t{j+1}=find(temp==j);
end
for i=1:size(t,2)
   yo(i)=size(t{i},2); % number of rewarding trials for each direction
end
%% Defining the bin size of data

i=0;
for j=1:size(t,2)
    for k=1:size(t{j},2)
        i=i+1;
        dat(i)=data(t{j}(k));
    end
end


dat(1).RnumSize=yo;
%% Removing units
ut=AfdStartOldRegress4_Per_neuron(dat,timebend,tlag,tlead,tdat)
ut(length(ut))=[];
utl=length(ut);
% ut=AfdStartOldRegress4_1_Per_neuron(FileName,ar,un,1000,1000)
%% important     -=---------------------------------------------------------------
% if(fornot>0)
    un=length(ut);
% end
clear data;
% ut=sort(ut);
data=dat;
for i=1:size(data,2)
    k=0;
    dat(i).spikes=[];
    for j=1:un %length(ut)% un%  % limit the number of units here
        k=k+1;
        if(j>length(ut))
            dat(i).spikes(k,:)=data(i).spikes(ut(randi([1 length(ut)],1,1)),:);
        else
            dat(i).spikes(k,:)=data(i).spikes(ut(j),:);
        end
    end
end
%% distancses
% for i=1:6
%     dat(1).sdis(i)=mean(ttr(:,i+1)-ttr(:,i))*50;
% end

%% experiment

% for i=1:size(dat,2)
%     dat(i).spikes=dat(i).spikes*(dat(i).Rnum+1);
%     if(dat(i).Rnum>0)
%         dat(i).spikes(size(dat(i).spikes,1)+1,:)=1;
%     else
%         dat(i).spikes(size(dat(i).spikes,1)+1,:)=0;
%     end
% %     dat(i).spikes(size(dat(i).spikes,1)+1,:)=dat(i).Rnum;
% end
% for i=1:size(pos,2)
% %     pos(i).spikes(size(pos(i).spikes,1)+1,:)=1;
%     pos(i).spikes=i*pos(i).spikes;
% end
%% experiment

%% collecting spike data
clear temp
for i=1:size(dat,2)
    for j=1:size(dat(i).spikes,2)/timebend
        for k=1:size(dat(i).spikes,1)
            temp(i).sp(k,j)=sum(dat(i).spikes(k,(j-1)*timebend+1:j*timebend));
        end
    end
end


for i=1:size(temp,2)
    n=0;
    for j=tlag:size(temp(i).sp,2)-tlead-1
        l=0;
        n=n+1;
        for m=1:size(temp(i).sp,1)
            for k=-tlag+1:tlead
                l=l+1;
                temp(i).spike(l,n)=temp(i).sp(m,j+k);
            end
        end
    end
end
%% concatanate for time
% k=0;
% for i=1:size(dat,2)
%     for j=1:size(dat(i).force,2)
%         k=k+1;
%         pos.spikes(:,k)=temp(i).spike(:,j);
%     end
% end


%% adding reward level to the input data Experiment
% for i=1:size(dat,2)
%     if(dat(i).Rnum>0)
%         temp(i).Rnum=1;
%     else
%         temp(i).Rnum=0;
%     end
% %     temp(i).Rnum=dat(i).Rnum;
%     temp(i).spike(size(temp(i).spike,1)+1,:)=temp(i).Rnum;
% end
%% collecting force data
k=0;
% i=i(randperm(length(i)));
for i=1:size(dat,2)%i(randperm(length(i)))  %=1:size(dat,2)
    for j=1:size(temp(i).spike,2)
        k=k+1;
        pos.spikes(:,k)=temp(i).spike(:,j);
        pos.f(k)=dat(i).force(j);
        pos.R(k)=dat(i).Rnum;
    end
end

%% temp
% p=[];
% p(1)=plot(pos.f)
% xlim([150 350]);
% xlabel('Time-->');
% ylabel('Force-->');
% title('Force value with time');
% hold on
% p(2)=plot(smoother(floor(pos.f*1),200,timebend))
% hold off
% legend(p,'Before Smoothing','After Smoothing');

%% temp
%% derivative of the force
if(derivative>0)
%     figure();
    plot(pos.f)
%     hold on
    temp=[]
    for i=2:length(pos.f)
        temp(i)=pos.f(i)-pos.f(i-1);
    end
    pos.f=temp;
%     plot(pos.f)
%     hold off
end
%% smoothing spike data
pos.spikes = smoother(pos.spikes, kks, timebend);
if(exsmooth>0)
    pos.f=smoother(floor(pos.f*1),kkf,timebend);
else
    pos.f=floor(pos.f*1);
end



%% adding ones on input
% pos.spikes(size(pos.spikes,1)+1,:)=ones(1,size(pos.spikes,2));
% pos.spikes(:,size(pos.spikes,2)+1)=ones(size(pos.spikes,1),1);

% for i=1:size(pos,2)
%     pos(i).spikes(size(pos(i).spikes,1)+1,:)=1;
% %     pos(i).spikes=i*pos(i).spikes;
% end

%% GP learning
N=floor(length(pos.f)*tdat);
for i=1:N
    train.f(i)=pos.f(i);
    train.spikes(:,i)=pos.spikes(:,i);
end
j=0;
for i=N+1:length(pos.f)
    j=j+1;
    test.f(j)=pos.f(i);
    test.spikes(:,j)=pos.spikes(:,i);
    test.R(j)=pos.R(j);
end
%% Gaussian Learning
sigma0 = std(train.f);
sigmaF0 = sigma0;
% d = size(train.spikes,1);
sigmaM0 = 0.5*sigma0;%10*ones(d,1);
Mfg=fitrgp(train.spikes',train.f','KernelFunction','squaredexponential',...
     'KernelParameters',[sigmaM0, sigmaF0],'Sigma',sigma0);
 
% Mfg=fitrgp(train.spikes',train.f');
test.pfg=(predict(Mfg,test.spikes'));
%% Linear Learning
% Mfg=fitrgp(train.spikes',train.f','Basis','constant','FitMethod','none',...
% 'PredictMethod','exact','KernelFunction','ardsquaredexponential',...
% 'KernelParameters',[sigmaM0;sigmaF0],'Sigma',sigma0,'Standardize',0);
% sigma0 = 0.1;
% kparams0 = [3.5, 6.2];

% Mfg=fitrgp(train.spikes',train.f','KernelFunction','squaredexponential',...
%      'Sigma',sigma0);
Mfl=fitlm(train.spikes',train.f');
test.pfl=(predict(Mfl,test.spikes'));

% Mf=train.f/train.spikes;
% test.pf=Mf*test.spikes;%-a;


fpcorg= corrcoef(test.f,test.pfg); %corrcoef(test.f,test.pfg)
fpcorl= corrcoef(test.f,test.pfl);%rsquare(test.f,test.pfl');
R(1)=fpcorl(2,1);
R(2)=fpcorg(2,1);
% Rsq=Mfl.Rsquared.Ordinary;
% test.px=smooth(test.px);
% test.py=smooth(test.py);
%% RSS calculation
    test.pfge=test.f-test.pfg';
    test.pfle=test.f-test.pfl';
    for j=1:length(test.pfge)
        test.pfge(j)=test.pfge(j)^2;
        test.pfle(j)=test.pfle(j)^2;
    end
    test.RSSg=sum(test.pfge);
    test.RSSl=sum(test.pfle);
Rss(1)=test.RSSl;
Rss(2)=test.RSSg;
%% figure for paper
clear p
close all
fact=figure('rend','painters','pos',[10 10 1200 800]);
hold on;
lenth=length(test.f);%98;
xax=1:lenth;
xax=xax/10;
p(1)=plot(xax,test.f(1:lenth),'-','Linewidth',6)%,test.y(1:length(test.px)));
p(2)=plot(xax,test.pfl(1:lenth),'-','Linewidth',6)%,test.py)
% p(3)=plot(test.pfg,'Linewidth',2)%,test.py)

set(gca,'fontsize',40)
% xlabel('Time Bins (100ms)','FontSize', 50);
% ylabel('Force (sensor reading)','FontSize', 50);
Mid=titlefunc(FileName)
title(strcat(Mid,' (Rl=',num2str(round(R(1)*100)/100),',n=',num2str(un),')'), 'FontSize', 40);%,' || Rg=',num2str(R(2)),')'))%
% title(strcat('Force Prediction (Rg=',num2str(fpcorg),'||Rl=',num2str(fpcorl),')'))%,' and MSE=',num2str(immse(test.y',test.py))));
% AX=legend(p,'Actual Force','Predicted Force');%,'Gaussian Model');
AX.FontSize = 30;
% set(gca,'fontsize',20)
%% figure the channel number

for i=1:length(ut)
    utch(i)=neural_data.spikeTimes{1,ar}{1,ut(i)}.ch;
end
utch=sort(utch);
[b1,m1,n1] = unique(utch,'first');
[c1,d1] =sort(m1);
clear utch;
utch= b1(d1);
%% save file
sv=0;
if(sv>0)
    i=1;
    a=[size(train.spikes,1) size(train.spikes,2)];
    sv=0;
    arr(1).name='-PMd';
    arr(2).name='-M1';
    arr(3).name='-S1';
    filename = strcat('C:\Users\matique\Google Drive\Houston-Lab\S1 Project\aaJournalS1\Figures\2. Mirror Force\',Mid,arr(ar).name,'.mat');
    info.utsort=sort(ut);
    info.ut=ut;
    info.utch=utch;
    info.tot=size(data(1).spikes,1);
    info.test=test;
    info.pos=pos;
    info.ar=ar;
    info.fdis=Mid;
    info.tlag=tlag;
    info.tlead=tlead;
    info.tdat=tdat;
    info.kkf=kkf;
    info.kks=kks;
    info.dat=dat;
    save(filename,'info')
end

%% save figure
if(sv>0)
    savefig(strcat(filename,'.fig'));
    saveas(fact,strcat(filename,'.png'));
end
% if(sv>0)
%     D=(area);
%     %filename = strcat('Dat',dat(1).p,'_Area',D,'_',dat(1).FileName);
%     savefig(strcat('Area_',ar(D).name,'_',FileName(11:32),'.fig'));
%     saveas(fact,strcat('Area_',ar(D).name,'_',FileName(11:32),'.jpg'));
%     %close all
% end
%% figures
% clear p
% close all
% fact=figure();
% hold on;
% p(1)=plot(test.f,'--','Linewidth',2)%,test.y(1:length(test.px)));
% p(2)=plot(test.pfl,':','Linewidth',2)%,test.py)
% % p(3)=plot(test.pfg,'Linewidth',2)%,test.py)
% 
% 
% xlabel('Time Bins (100ms) -->', 'FontSize', 15);
% ylabel('Force (sensor reading) -->', 'FontSize', 15);
% Mid=titlefunc(FileName)
% title(strcat(Mid,' (Rl=',num2str(R(1)),')'));%,' || Rg=',num2str(R(2)),')'))%
% % title(strcat('Force Prediction (Rg=',num2str(fpcorg),'||Rl=',num2str(fpcorl),')'))%,' and MSE=',num2str(immse(test.y',test.py))));
% legend(p,'Actual Force','Linear Model');%,'Gaussian Model');
% 
% i=1;
%% dorkar nai eita
% ch=test.R(i);
% text(i,400,num2str(test.R(i)),'FontSize', 20)
% for i=2:length(test.R)
%     if(ch==test.R(i))
%     else
%         ch=test.R(i);
%         text(i,400,num2str(test.R(i)),'FontSize', 20)
%     end
% end
% hold off


%% saving data files
% dat(1).state=state;
% dat(1).FileName=FileName;
% dat(1).area=Flag.area;
% dat(1).p=p;
% dat(1).prevt(1)=prevt*50;
% dat(1).postt(1)=postt*50;
% filename = 'data.mat';
% save(filename,'dat')
