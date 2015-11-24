clear

% load workspace
run = 4; % run 1: 1 robot; run 3: 4 robots, 3 working; 
ws_filename = strcat('./workspaces/run_', int2str(run),'_vars_data.mat')
load(ws_filename)

figure_handle = figure(1)

%tmax = 900
tmax = size(T,2)
plot_arena(figure_handle, task_sites, boats_pos(:,1:tmax,:), squeeze(boats_task(:,1:tmax,:)), 0);


