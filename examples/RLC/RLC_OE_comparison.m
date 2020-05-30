clear;
clc;

%% Read data table 

data_path = fullfile("data", "RLC_data_id.csv");
data_table = readtable(data_path);


vin =  data_table.V_IN;
vC = data_table.V_C;
iL = data_table.I_L;
y = [vC iL];
t = data_table.time;
Ts = t(2) - t(1);

%% Add noise %%
add_noise = 0;
STD_V = add_noise*10;
STD_I = add_noise*1;
vC_meas = vC + randn(size(vC))*STD_V;
iL_meas = iL + randn(size(iL))*STD_V;
y_meas = [vC_meas iL_meas];

%% Identification data %%
data_id = iddata(y_meas,vin,Ts);
model_subs = oe(data_id, 'nb',[2; 2], 'nf', [2; 2]);

y_sim_id = sim(model_subs, data_id);
y_sim_id = y_sim_id.OutputData;


%% Plot data %%

figure()
plot(t, vC, 'k');
hold on;
plot(t, y_sim_id(:,1), 'b');
legend('True', 'Model');

figure()
plot(t, iL, 'k');
hold on;
plot(t, y_sim_id(:,2), 'b');
legend('True', 'Model');

%%
SSE_v = sum((vC - y_sim_id(:,1)).^2);
y_mean_v = mean(vC);
SST_v = sum((vC - y_mean_v).^2);
R_sq_v = 1 - SSE_v/SST_v;

SSE_i = sum((iL - y_sim_id(:,2)).^2);
y_mean_i = mean(iL);
SST_i = sum((iL - y_mean_i).^2);
R_sq_i = 1 - SSE_i/SST_i;

fprintf("OE fitting performance");
fprintf("Identification dataset:\nR-squred vC:%.3f\nR-squred iL:%.3f\n", R_sq_v, R_sq_i)

%% Read data table val

data_path = fullfile("data", "RLC_data_val.csv");
data_table_val = readtable(data_path);


vin =  data_table_val.V_IN;
vC = data_table_val.V_C;
iL = data_table_val.I_L;
y = [vC iL];
t = data_table.time;
Ts = t(2) - t(1);

%% Validation data %%
data_val = iddata(y_meas,vin,Ts);

y_sim_val = sim(model_subs, data_val);
y_sim_val = y_sim_val.OutputData;

loss = mean((vC - y_sim_val).^2);

%%
SSE_v = sum((vC - y_sim_val(:,1)).^2);
y_mean_v = mean(vC);
SST_v = sum((vC - y_mean_v).^2);
R_sq_v = 1 - SSE_v/SST_v;

SSE_i = sum((iL - y_sim_val(:,2)).^2);
y_mean_i = mean(iL);
SST_i = sum((iL - y_mean_i).^2);
R_sq_i = 1 - SSE_i/SST_i;

fprintf("Validation dataset:\nR-squred vC:%.2f\nR-squred iL:%.2f\n", R_sq_v, R_sq_i)
