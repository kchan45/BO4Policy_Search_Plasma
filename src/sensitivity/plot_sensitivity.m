clear; clc; close all

% setup global settings for plots
set(0, 'DefaultLineMarkerSize', 8);
set(0, 'DefaultLineLineWidth', 2);
set(0, 'DefaultAxesFontSize', 15);

colors = {'#0072BD', '#7E2F8E', '#A2142F', '#77AC30'};
colororder(colors)

wksp_data = load('./2022_09_27_14h00m00s_sensitivity_workspace_nsamp10000.mat');

sens = wksp_data.sens;
n_param_first = length(wksp_data.bias_1_row) + length(wksp_data.weights_1_row);
n_param_last = length(wksp_data.bias_end_row) + length(wksp_data.weights_end_row);

first_layer_sens = sens(1:n_param_first,:);
last_layer_sens = sens(n_param_first+1:end,:);

first_layer_params = 1:n_param_first;
last_layer_params = n_param_first+[1:n_param_last];

mean_first_layer_sens = mean(first_layer_sens);
mean_last_layer_sens = mean(last_layer_sens);
std_first_layer_sens = std(first_layer_sens);
std_last_layer_sens = std(last_layer_sens);
ste_first_layer_sens = std_first_layer_sens/sqrt(n_param_first);
ste_last_layer_sens = std_last_layer_sens/sqrt(n_param_last);

line_prop = 'o:';

plot(first_layer_params, sens(first_layer_params,1), line_prop, 'Color', colors{1})
hold on
plot(first_layer_params, sens(first_layer_params,2), line_prop, 'Color', colors{2})
plot(last_layer_params, sens(last_layer_params,1), line_prop, 'Color', colors{3})
plot(last_layer_params, sens(last_layer_params,2), line_prop, 'Color', colors{4})

for i = 1:length(mean_first_layer_sens)
val = mean_first_layer_sens(i);
plot([1,n_param_first], [val,val] ,'--', 'Color', colors{i})
end

for i = 1:length(mean_last_layer_sens)
val = mean_last_layer_sens(i);
plot(n_param_first+[1,n_param_last], [val,val] ,'--', 'Color', colors{i+2})
end

xlabel('Parameter Number')
ylabel('Sensitivity Value')
