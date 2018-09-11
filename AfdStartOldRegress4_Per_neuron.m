function [ut p]=AfdStartOldRegress4_Per_neuron(dat,timebend,tlag,tlead,tdat)
%% collecting spike data
%% info
% for time lag and lead
%% Info
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
k=0;
for i=1:size(dat,2)
    for j=1:size(dat(i).force,2)
        k=k+1;
        pos.spikes(:,k)=temp(i).spike(:,j);
    end
end

%% collecting force data
k=0;
for i=1:size(dat,2)
    for j=1:size(temp(i).spike,2)
        k=k+1;
        pos.f(k)=dat(i).force(j);
    end
end

%% smoothing spike data
exsmooth=1;
pos.spikes = smoother(pos.spikes, 20, timebend);
if(exsmooth>0)
    pos.f=smoother(floor(pos.f*1),50,timebend);
else
    pos.f=floor(pos.f*1);
end

%% Dividing Training and Testing set
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
end
%% Gaussian model
% sigma0 = std(train.f);
% sigmaF0 = sigma0;
% sigmaM0 = 0.5*sigma0;%10*ones(d,1);
% Mfg=fitrgp(train.spikes',train.f','KernelFunction','squaredexponential',...
%      'KernelParameters',[sigmaM0, sigmaF0],'Sigma',sigma0);
%  
% test.pfg=(predict(Mfg,test.spikes'));
% 
% %% Linear models
% Mfl=fitlm(train.spikes',train.f');
% test.pfl=(predict(Mfl,test.spikes'));
% 
% Mf=train.f/train.spikes;
% test.pf=Mf*test.spikes;%-a;

%% checking correlation for each unit
k=0;
tT=tlag+tlead;
for i=1:size(train.spikes,1)/tT
    k=k+1;
    test.sp(1:tT,:)=test.spikes((i-1)*tT+1:i*tT,:);
    train.sp(1:tT,:)=train.spikes((i-1)*tT+1:i*tT,:);
    clear Mdl
    Mdl=fitlm(train.sp',train.f');
    test.pflu=(predict(Mdl,test.sp'));
    fpcorl= corrcoef(test.f,test.pflu);%rsquare(test.f,test.pfl');
    %[util(i,1) util(i,2)]=ranksum(test.f,test.pflu);
    util(i)=fpcorl(2,1);%Mdl.Rsquared.Ordinary;
   
    stat=anova(Mdl,'summary');
    stat=table2array(stat);
    pval(i)=stat(2,5);
end
%% sorting the unit based Corr coeff
clear tmp
util(isnan(util))=0;
tmp=util;
for i=1:length(util)
    tt=[];
    tt=find(tmp==max(tmp));
    a(i)=tt(1);
    tmp(a(i))=-50;
end
%% Adding or pruning the unit
k=0;
for i=1:length(a)
    k=k+1;
    ut(k)=a(i);
    for j=1:length(ut)
        test.sp((j-1)*tT+1:j*tT,:)=test.spikes((ut(j)-1)*tT+1:ut(j)*tT,:);
        train.sp((j-1)*tT+1:j*tT,:)=train.spikes((ut(j)-1)*tT+1:ut(j)*tT,:);
    end
    clear Mdl
    Mdl=fitlm(train.sp',train.f');
    test.pflu=(predict(Mdl,test.sp'));
    fpcorl= corrcoef(test.f,test.pflu);
    if(i==1)
        chk=fpcorl(2,1);
    else
        if(chk<fpcorl(2,1))
            chk=fpcorl(2,1);
        else
            k=k-1;
        end
    end
    R(1)=fpcorl(2,1);
end

%% Gaussian model
% sigma0 = std(train.f);
% sigmaF0 = sigma0;
% sigmaM0 = 0.5*sigma0;%tlag*ones(d,1);
% Mfg=fitrgp(train.sp',train.f','KernelFunction','squaredexponential',...
%      'KernelParameters',[sigmaM0, sigmaF0],'Sigma',sigma0);
%  
% test.pfg=(predict(Mfg,test.sp'));
% fpcorg= corrcoef(test.f,test.pfg); %corrcoef(test.f,test.pfg)
% % fpcorl= corrcoef(test.f,test.pfl);%rsquare(test.f,test.pfl');
% R(2)=fpcorg(2,1);
% % R(1)=fpcorl(2,1);
% % R(2)=fpcorg(2,1)
% % Rsq=Mfl.Rsquared.Ordinary;
%% figures
% clear p
% close all
% fact=figure()
% hold on;
% p(1)=plot(test.f,'--','Linewidth',2)%,test.y(1:length(test.px)));
% p(2)=plot(test.pflu,':','Linewidth',2)%,test.py)
% p(3)=plot(test.pfg,'Linewidth',2)%,test.py)
% hold off
% xlabel('time -->', 'FontSize', 15);
% ylabel('Force -->', 'FontSize', 15);
% title(strcat('Force Prediction (Rl=',num2str(R(1)),' || Rg=',num2str(R(2)),')'))%
% % title(strcat('Force Prediction (Rg=',num2str(fpcorg),'||Rl=',num2str(fpcorl),')'))%,' and MSE=',num2str(immse(test.y',test.py))));
% legend(p,'Actual Force','Linear Model','Gaussian Model');
% sv=0;
% clear ar
% ar(1).name='PMd';
% ar(2).name='M1';
% ar(3).name='S1';
% if(sv>0)
%     D=(area);
%     %filename = strcat('Dat',dat(1).p,'_Area',D,'_',dat(1).FileName);
%     savefig(strcat('Area_',ar(D).name,'_',FileName(11:32),'.fig'));
%     saveas(fact,strcat('Area_',ar(D).name,'_',FileName(11:32),'.jpg'));
%     %close all
% end

