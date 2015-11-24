% Simple code that runs boats in simulation and real

% 1) set boat IDs
% 2) check Ks available for ntasks
% 3) set boats initial species and tasks
% 4) set max time
% 5) set run number
% 6) set fix_Ks to false if real optim
% 7) set nspecies

%% Setup

clear 
file_time = datestr(clock);

% 5: 1 task, 4 robs
% 6: 4 tasks, 4  robs, failed at end
% 7: 4 tasks, 4  robs, S=2
% 8: 4 tasks, 4  robs, S=2, changed init pos to be spread out over 1 and 2
% 9: fixed Ks, transposing; same as run 8
% 10: longer, less avoidance, same as 9
% 11: no avoidance
% 12:
run = 29;

save_data = true;
ws_filename = strcat('run_',int2str(run),'_all_data.mat'); %strcat(file_time,'_data.mat');
vars_filename = strcat('run_',int2str(run),'_vars_data.mat'); %strcat(file_time,'_pos.mat');
 
% Real or simulated.
simulate = false;
make_movie = false;
fix_Ks = false;
% which case?
N4Q1 = 1;
N4Q2 = 0;
N7Q1 = 0;
N6Q1 = 0;
N41Q1 = 0;
N41Q2 = 0;

% Constants.
max_time = 400;  % in seconds.
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

% initialize boats
nboats = 1; % overloaded if ~simulate
nspecies = 2;
ntasks = 4; % currently works for 1,2,3,4
if ~simulate
    % Setup boats.
    %boat_ids = [101, 102, 103, 104, 105]
    %boat_ids = [104, 101, 106, 103, 105, 112, 113];
    %boat_hc_ids = [4, 1, 6, 3, 5, 12, 13];
    boat_ids = [104, 101, 103, 102];
    boat_hc_ids = [4, 1, 3, 2];
    nboats = length(boat_ids)
end

if ~simulate
    % communication with mASV
    try
        fclose(port)
    end
    global port;
    port = serial('COM4', 'baudrate', 57600, 'Terminator', '');
    fopen(port);
    % communication with OptiTrack
    try
        getPose2d_filt stop
    end
    getPose2d_filt init

    for i = 1:nboats
        boats{i} = struct();
        boats{i}.ID = boat_ids(i)  % opti-track ID: 100+x
        boats{i}.boatID = boat_hc_ids(i); % Boat hard-coded ID: x
    end

    % Boat control parameters.
    global th_cmd_max; global vel_cmd_max; global th_gain; global vel_gain;
    th_cmd_max = 150;   % saturation value for heading command, to prevent crazy swings.
    vel_cmd_max = 225;  % saturation value for velocity command.
    th_gain = struct('P', 300, 'I', 0, 'D', 100); % 300   0  100
    vel_gain = struct('P', 3300, 'I', 0, 'D', 0); % 3300  0  0
end

%% Initialize values

% Optimized K matrices (to be taken from the Python code).
% Rates must be small must be low (in the order of a boat make 1-2 turns around the task site).
K = cell(nspecies);
if fix_Ks
    if ntasks==4
        K{1} = [-0.01, 0.0033, 0.0033, 0.0033; 0.0033, -0.01, 0.0033, 0.0033; 0.0033, 0.0033, -0.01, 0.0033;0.0033,0.0033,0.0033,-0.01];
        K{2} = [-0.01, 0.0033, 0.0033, 0.0033; 0.0033, -0.01, 0.0033, 0.0033; 0.0033, 0.0033, -0.01, 0.0033;0.0033,0.0033,0.0033,-0.01];
    elseif ntasks==3
        K{1} = [-0.01, 0.005, 0.005; 0.005, -0.01, 0.005; 0.005, 0.005, -0.01];
        K{2} = [-0.01, 0.005, 0.005; 0.005, -0.01, 0.005; 0.005, 0.005, -0.01];
    elseif ntasks==2
        K{1} = [-0.01, 0.01,; 0.01, -0.01];
        K{2} = [-0.01, 0.01; 0.01, -0.01];
    end
else % 2 species, 4 tasks; from init 1,2 to final 3,4
    if N4Q1
    %  Q is complementary; N=4
    K{1} = [-0.02 0.0 0.0 0.0; ...
        0.01 -0.01 0.0 0.0; ...
        0.01 0.0 -0.01 0.0; ...
        0.0 0.01 0.01 -0.0];
    K{2} = [-0.01 0.01 0.0 0.0; ...
        0.0 -0.02 0.0 0.0; ...
        0.01 0.0 -0.0 0.01; ...
        0.0 0.01 0.0 -0.01];
    end
    if N4Q2
    % Q is redundant; N=4
    K{1} = [-0.0193031387543 0.00929474771883 0.0 0.0; ...
        0.00930313875426 -0.0192947477188 0.0 0.0; ...
        0.01 0.0 -0.00799835383383 0.00840937795305; ...
        0.0 0.01 0.00799835383383 -0.00840937795305];
    K{2} = [-0.0179431595788 0.00796603329173 0.0 0.0; ...
        0.0079431595788 -0.0179660332917 0.0 0.0; ...
        0.01 0.0 -0.00886413867022 0.0084308753447; ...
        0.0 0.01 0.00886413867022 -0.0084308753447];
    end
    if N7Q1
        K{1} = [-0.02 0.0 0.0 0.0; ...
        0.01 -0.01 0.0 0.0; ...
        0.01 0.0 -0.01 0.0; ...
        0.0 0.01 0.01 -0.0];
        K{2} = [-0.01 0.00970463237956 0.0 0.0; ...
        0.0 -0.0197046323796 0.0 0.0; ...
        0.01 0.0 -0.0 0.01; ...
        0.0 0.01 0.0 -0.01];

    end
    if N41Q1
    K{1} = [-0.02 0 0.0 0.0; ...
        0.01 -0.010 0.0 0.0; ...
        0.01 0.0 -0.01 0.0; ...
        0.0 0.01 0.01 -0.0];
K{2} = [-0.0106009541946 0.01 0.0 0.0; ...
        0.000600954194627 -0.018087591233 0.0 0.0; ...
        0.01 0.0 -0.0 0.0085620560548; ...
        0.0 0.00808759123296 0.0 -0.0085620560548];
    end
    if N41Q2

K{1} = [-0.02 0.00175466480543 0.0 0.0; ...
        0.01 -0.0117546648054 0.0 0.0; ...
        0.01 0.0 -0.01 0.00327847760618; ...
        0.0 0.01 0.01 -0.00327847760618];
K{2} = [-0.018334612209 0.0 0.0 0.0; ...
        0.00844838046167 -0.01 0.0 0.0; ...
        0.00988623174729 0.0 -0.00874322218269 0.0; ...
        0.0 0.01 0.00874322218269 -0.0];
    end
    if N6Q1
        K{1} = [-0.02 3.13699471898e-06 1.178271887e-05 0.0; ...
        0.01 -0.0100031369947 0.0 0.0; ...
        0.01 0.0 -0.0100117827189 0.0; ...
        0.0 0.01 0.01 -0.0];
K{2} = [-0.0100476568277 0.01 0.0 0.0; ...
        4.76568276985e-05 -0.02 0.0 0.0; ...
        0.01 0.0 -0.0 0.0099941501237; ...
        0.0 0.01 0.0 -0.0099941501237];

    end
end


P = cell(nspecies);
for i = 1:nspecies
    P{i} = expm(K{i} * dt);
end

% Give species to each boat.
if simulate
    boats_species = randi(nspecies, nboats, 1)
else
    % When controlling the ACTUAL boats, grab set of constant species instead.
    % error('Setting species on real boats not implemented.');
    % Set fixed species for real-robot runs, for example:
    disp('Setting up species for 1 boat');
    boats_species = zeros(nboats, 1);
    boats_species(1) = 1;
    boats_species(2) = 1;
    boats_species(3) = 2;
    boats_species(4) = 2;
    %boats_species(5) = 2;
    %boats_species(6) = 2;
    %boats_species(7) = 2;
end

% Setup task sites.
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


% Places N boats randomly (within 3x3 arena).
if simulate
    boats_init_pos = rand(nboats, 2) * arena_size;
else
    % When controlling the ACTUAL boats, grab the initial positions of the boats instead.
    boats_init_pos = zeros(nboats, 2);
    for i = 1:nboats
        boats{i} = update_boatState(boats{i});
        boats_init_pos(i, 1) = boats{i}.x;
        boats_init_pos(i, 2) = boats{i}.y;
    end
end

% Assign tasks to boats.
if simulate
    boats_init_task = randi(ntasks, nboats, 1);
else
    % When controlling the ACTUAL boats, grab set of constant initial task distribution instead.
    %error('Setting initial tasks for real boats not implemented.');
    disp('Setting up initial tasks for boats')
    % Set a fixed init distrib for real-robot runs, for example:
    boats_init_task = zeros(nboats, 1);
    boats_init_task(1) = 1;
    boats_init_task(2) = 1;
    boats_init_task(3) = 2;
    boats_init_task(4) = 2;
    %boats_init_task(5) = 2;
    %boats_init_task(6) = 2;
    %boats_init_task(7) = 2;
end

% Trajectories of the boats.
boats_pos = zeros(nboats, length(T), 2);
boats_task = zeros(nboats, length(T), 1);
for i = 1:nboats
    boats_pos(i, 1, :) = boats_init_pos(i, :);
    boats_task(i, 1, :) = boats_init_task(i, :);
end


%% Main Loop

tic;
i = 1;
while i < size(T, 2)
   if ~simulate && toc < dt
       continue;
   end
   tic;
   i = i + 1;
   if(mod(i,50)==0)
    txt = sprintf('%d / %d',i,size(T, 2));
    disp(txt);
   end
   
    % Update position.
    if ~simulate
        for j = 1:nboats % grab position from MoCap
            boats{j} = update_boatState(boats{j});
            boats_pos(j, i - 1, 1) = boats{j}.x;
            boats_pos(j, i - 1, 2) = boats{j}.y;
        end
    end

    t = T(i);
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
        if simulate
            npos = pos + dx * dt;
            boats_pos(j, i, :) = npos;
        else
            % When controlling the ACTUAL boats, set the speed of the boats instead.
            boats{j}.vel_des_vect = dx;
            % Compute velocity commands for the boat.
            boats{j} = computeBoatCmd(boats{j});
            % Set boat motors with the computed values (must be executed twice).
            setMotors(port, boats{j}.boatID, boats{j}.M1_cmd, boats{j}.M2_cmd);
            setMotors(port, boats{j}.boatID, boats{j}.M1_cmd, boats{j}.M2_cmd);
        
        end

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

%% Stop experiment

% stop experiment: grab last position and set to zero
if ~simulate
    for j = 1:nboats
        boats{j} = update_boatState(boats{j});
        boats_pos(j, length(T), 1) = boats{j}.x;
        boats_pos(j, length(T), 2) = boats{j}.y;
        setMotors(port, boats{j}.boatID, 0, 0);
        setMotors(port, boats{j}.boatID, 0, 0);
    end
    fclose(port);
    delete(port);
    clear port;
    disp('Closed port')
end

%% Save data

% save workspace
if save_data
    save(ws_filename);
    save(vars_filename,'boats','boats_pos','task_sites','boats_task','T', 'dt', 'boats_species');
end

figure_handle = figure(1);
plot_arena(figure_handle, task_sites, boats_pos, squeeze(boats_task(:,end,:)), make_movie);

