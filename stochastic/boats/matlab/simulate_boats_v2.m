% Simple code that runs boats in simulation
% Some values are hardcoded:
% species_traits
% sum_species (#robots per species)


%% Setup

clear 
file_time = datestr(clock);

run = 1;

save_data = true;
ws_filename = strcat('run_',int2str(run),'_all_data.mat'); %strcat(file_time,'_data.mat');
vars_filename = strcat('run_',int2str(run),'_vars_data.mat'); %strcat(file_time,'_pos.mat');
 
% Real or simulated.
make_movie = false;
fix_Ks = true;

% Constants.
max_time = 100;  % in seconds.
dt = 0.2;
T = 0:dt:max_time;

setup_time = 60;  % Time during which no task switching is allowed (<= max_time).
velocity_on_circle = 0.06;
avoidance_velocity = 0.002;
avoidance_range = -10; % no avoidance 
min_velocity = 0.03;
max_velocity = 0.08;
task_radius = 0.25;
arena_size = 3;


%% Initialize boats

nboats = 100;
nspecies = 3;
ntasks = 10; % currently works for 1,2,3,4
sum_species = [30,10,60];

% Create species
s1 = ones(sum_species(1),1) * 1;
s2 = ones(sum_species(2),1) * 2;
s3 = ones(sum_species(3),1) * 3;
temp_species = [s1;s2;s3];
ri = randperm(nboats);
boats_species = temp_species(ri);

% Give species to each boat.
%boats_species = randi(nspecies, nboats, 1)
% Places N boats randomly (within 3x3 arena).
boats_init_pos = rand(nboats, 2) * arena_size;
% Assign tasks to boats.
boats_init_task = randi(ntasks, nboats, 1);

% Trajectories of the boats.
boats_pos = zeros(nboats, length(T), 2);
boats_task = zeros(nboats, length(T), 1);
for i = 1:nboats
    boats_pos(i, 1, :) = boats_init_pos(i, :);
    boats_task(i, 1, :) = boats_init_task(i, :);
end


%% Initialize transition values

% Optimized K matrices (to be taken from the Python code).
% Rates must be small must be low (in the order of a boat make 1-2 turns around the task site).


nspecies = 3
K = cell(nspecies);
K{1} = [-1.16893832598 0.0 0.0 0.0 0.0 0.0 0.705684493218 0.0 0.0 0.00040998642805; ...
        0.0 -0.173875400981 0.0 0.0 0.298153566145 0.0 0.0 0.0 0.0 0.0; ...
        0.0 0.0 -1.0 0.0370109635236 0.0 0.0 0.0 0.0 0.0 0.0; ...
        0.0 0.0 1.0 -1.03701096352 0.619343707061 0.0 0.0 0.0 0.0 0.0; ...
        0.0 0.173875400981 0.0 1.0 -1.91749727321 0.542103915027 0.0 0.0 0.0 0.0; ...
        0.0 0.0 0.0 0.0 1.0 -1.54210391503 1.0 0.0 0.0 0.0; ...
        0.976937934426 0.0 0.0 0.0 0.0 1.0 -2.70568449322 0.747833937072 0.0 0.0; ...
        0.0 0.0 0.0 0.0 0.0 0.0 1.0 -0.752207991507 0.178511155316 0.0; ...
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0043740544342 -1.14964220739 0.454002063846; ...
        0.192000391558 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.971131052071 -0.454412050274];
K{2} = [-1.39108633774 0.0 0.0 0.0 0.0 0.0 0.765135115199 0.0 0.0 0.71437632878; ...
        0.0 -1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; ...
        0.0 0.0 -0.00444896625698 0.129962399594 0.0 0.0 0.0 0.0 0.0 0.0; ...
        0.0 0.0 0.00444896625698 -0.980966475528 0.00660987252861 0.0 0.0 0.0 0.0 0.0; ...
        0.0 1.0 0.0 0.851004075934 -1.00660987253 0.0 0.0 0.0 0.0 0.0; ...
        0.0 0.0 0.0 0.0 1.0 -1.0 1.0 0.0 0.0 0.0; ...
        0.606642350011 0.0 0.0 0.0 0.0 1.0 -2.40628755613 0.79890028998 0.0 0.0; ...
        0.0 0.0 0.0 0.0 0.0 0.0 0.641152440935 -0.808512927911 1.0 0.0; ...
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.00961263793093 -1.0 0.649907899232; ...
        0.784443987732 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -1.36428422801];
K{3} = [-1.63298673504 0.0 0.0 0.0 0.0 0.0 0.790444290793 0.0 0.0 0.0; ...
        0.0 -0.407555398833 0.0 0.0 0.999450962344 0.0 0.0 0.0 0.0 0.0; ...
        0.0 0.0 -0.339758818269 0.445139829123 0.0 0.0 0.0 0.0 0.0 0.0; ...
        0.0 0.0 0.339758818269 -0.476125381568 0.0599066384972 0.0 0.0 0.0 0.0 0.0; ...
        0.0 0.407555398833 0.0 0.0309855524455 -1.17539246652 0.40542979888 0.0 0.0 0.0 0.0; ...
        0.0 0.0 0.0 0.0 0.116034865675 -1.40542979888 0.622151391643 0.0 0.0 0.0; ...
        0.632986735039 0.0 0.0 0.0 0.0 1.0 -2.41259568244 0.32875568158 0.0 0.0; ...
        0.0 0.0 0.0 0.0 0.0 0.0 1.0 -1.31965418311 0.666935501628 0.0; ...
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.990898501531 -0.996695826024 1.0; ...
        1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.329760324397 -1.0];



P = cell(nspecies);
for i = 1:nspecies
    P{i} = expm(K{i} * dt);
end


%% Setup task sites.

task_sites = zeros(ntasks, 2);
if ntasks == 1
    % hard code 1st task site
    task_sites(1, :) = [1.5 1.5];
else
    if ntasks == 4
        % CW from top left: 1-3-4-2
        o = 0.65;
        c = 1.5;
        task_sites(1, :) = [c-o, c+o];
        task_sites(2, :) = [c-o, c-o];
        task_sites(3, :) = [c+o, c+o];
        task_sites(4, :) = [c+o, c-o];
    else
        for i = 1:ntasks
            a = (i - 1) / ntasks * 2 * pi;
            task_sites(i, :) = [cos(a), sin(a)] * (arena_size / 2.8 - task_radius - 0.1) + arena_size / 2;
        end
    end
end



%% Main Loop

tic;
i = 1;
while i < size(T, 2)
  
    tic;
    i = i + 1;
    if(mod(i,50)==0)
        txt = sprintf('%d / %d',i,size(T, 2));
        disp(txt);
    end
    t = T(i);
    % Update position.

    for j = 1:nboats
        task = boats_task(j, i - 1, 1);
        task_center = task_sites(task, :);
        pos = squeeze(boats_pos(j, i - 1, :));
        obstacles = zeros(nboats - 1, 2); % for each boat, pos of all other boats
        for k = 1:nboats-1
            offset = 0;
            if k >= j
                offset = 1;
            end
            obstacles(k, :) = boats_pos(k + offset, i - 1, :);
        end
        % vector dx contains vel in x and y directions
        dx = compute_velocity(pos, task_radius, task_center, obstacles, velocity_on_circle, avoidance_velocity, avoidance_range);
        v = norm(dx);
        if v > max_velocity
            dx = dx / v * max_velocity;
        elseif v < min_velocity
            dx = dx / v * min_velocity;
        end

        % Update position using Euler integration.
        npos = pos + dx * dt;
        boats_pos(j, i, :) = npos;


        % Switch to another task?
        if t > setup_time
            if ntasks > 1
                if ~fix_Ks
                    ntask = randsample(1:ntasks, 1, true, P{boats_species(j)}(:, task));
                else
                    ntask = randsample(1:ntasks, 1, true, P{boats_species(j)}(task, :));
                end
            else
                ntask = 1;
            end
            if ntask ~= task
                temp = sprintf('Boat %d switched to task %d\n', j, ntask);
                disp(temp)
            end
        else
            ntask = task;
        end
        boats_task(j, i) = ntask;
    end

 
end


%% Save data

% save workspace
if save_data
    save(ws_filename);
    save(vars_filename,'boats_pos','task_sites','boats_task','T', 'dt', 'boats_species');
end

figure_handle = figure(1);
plot_arena(figure_handle, task_sites, boats_pos, squeeze(boats_task(:,end,:)), make_movie);

