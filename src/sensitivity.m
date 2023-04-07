clearvars
clc
uqlab

load('./saved/mat_files/2022_09_16_13h47m02s_initial_policy_info.mat')
Nsamples = 10000; %We need a lot for S.A.

weights_1_row = reshape(W{1}',1,[]) ;%Flattened weights, first layer, row first
bias_1_row = reshape(b{1}',1,[]) ;%Flattened biases, first layer, row first
weights_end_row = reshape(W{end}',1,[]) ;
bias_end_row = reshape(b{end}',1,[]) ;

nominal_values = horzcat(weights_1_row,...
    bias_1_row,weights_end_row,bias_end_row)';

%Gather bounds from which we want to sample
% (assuming uniform distribution)
sensitivity_bounds = zeros(44,2);
r = 2; %Adopting "geometric" bounds, could be a plus/minus as well

for j=1:length(sensitivity_bounds)

if nominal_values(j)>0
    sensitivity_bounds(j,1)=(1/r)*nominal_values(j);
    sensitivity_bounds(j,2)=(r)*nominal_values(j);
else
    sensitivity_bounds(j,1)=(r)*nominal_values(j);
    sensitivity_bounds(j,2)=(1/r)*nominal_values(j);
end
end


for k=1:length(sensitivity_bounds)
inputOpts.Marginals(k).Type = 'Uniform';
inputOpts.Marginals(k).Parameters= [sensitivity_bounds(k,1),...
                            sensitivity_bounds(k,2)];
end

myInput = uq_createInput(inputOpts); %Create input object
X = uq_getSample(Nsamples,'LHS');

%Enumerate elements in total flattened matrix X
%to gather for sensitivity

w1e = 1:size(W{1},1)*size(W{1},2);
b1e = (w1e(end)+1): (w1e(end)+size(b{1},2));
wEe = b1e(end)+1:b1e(end)+size(W{end},1)*size(W{end},2);
bEe = wEe(end)+1:wEe(end)+size(b{end},2);

%Get back the tensor form to input in DNN
W1   = reshape(X(:,w1e), [Nsamples,size(W{1},1),size(W{1},2)]);
B1   = reshape(X(:,b1e), [Nsamples,size(b{1},1),size(b{1},2)]);
Wend = reshape(X(:,wEe)',[Nsamples,size(W{end},1),size(W{end},2)]);
Bend = reshape(X(:,bEe), [Nsamples,size(b{end},1),size(b{end},2)]);

Y = compute_objectives(W1,B1,Wend,Bend,Nsamples);


BorgonovoOpts.Type = 'Sensitivity';
BorgonovoOpts.Method = 'Borgonovo';
BorgonovoOpts.Borgonovo.Method = 'HistBased';

BorgonovoOpts.Borgonovo.Sample.X = X;
BorgonovoOpts.Borgonovo.Sample.Y = Y;
BorgonovoAnalysis= uq_createAnalysis(BorgonovoOpts);
sens = BorgonovoAnalysis.Results.Delta; %Delta are the indices
