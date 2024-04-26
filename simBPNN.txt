function [ytrp, ytep, ytr, yte, options] = simBPNN(trainD,testD) 

% normalization
options.normalization=0;
if options.normalization==1;

data=[trainD;testD];
minb=min(data);
maxb=max(data);
xmin=minb(1,1:end-1);    xmax=maxb(1,1:end-1);

for j=1:size(trainD,1)
    trainD(j,1:end-1) = normalizer(trainD(j,1:end-1),xmin, xmax, 1); 
end
for j=1:size(testD,1)
    testD(j,1:end-1) = normalizer(testD(j,1:end-1),xmin, xmax, 1);
end
end

xtr=trainD(:,1:end-1);
ytr=trainD(:,end);
xte=testD(:,1:end-1);
yte=testD(:,end);
options.xtr=xtr;
options.ytr=ytr;

net=newff(xtr',ytr',size(xtr,2));

% Set the parameters
net.trainFcn ='trainrp';
net.trainParam.epochs=1000;
net = train(net,xtr',ytr');

temp1 = sim(net,xtr');
temp2 = sim(net,xte');
ytrp=temp1';ytep=temp2';