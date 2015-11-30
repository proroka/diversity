% Simple code that runs boats in simulation
% Some values are hardcoded:
% species_traits
% sum_species (#robots per species)


%% Setup

clear
file_time = datestr(clock);

run = 8;

% from the trait evolution data, choose which time slot [1,2,3]
slot = 1;
% If true, the simulation will cycle through the different slots.
nslots = 3;
auto_advance = true;

save_data = true;
ws_filename = strcat('run_',int2str(run),'_all_data.mat'); %strcat(file_time,'_data.mat');
vars_filename = strcat('run_',int2str(run),'_vars_data.mat'); %strcat(file_time,'_pos.mat');

make_movie = false;
plot_on = false;

% Constants.
setup_time = 60;  % Time during which no task switching is allowed (<= max_time).
time_per_slot = 700
if auto_advance
    max_time = 60 + time_per_slot * nslots;  % in seconds.
else
    max_time = 60 + time_per_slot
end
dt = 2.;
T = 0:dt:max_time;
dt_per_slot = round(time_per_slot / dt)
dt_for_setup = round(setup_time / dt)



velocity_on_circle = 0.06;
avoidance_velocity = 0.002;
avoidance_range = -10; % no avoidance
min_velocity = 0.03;
max_velocity = 0.08;
task_radius = 0.05;
arena_size = 3;


%% Initialize boats

% these values are hardcoded in python code
nboats = 100;
nspecies = 3;
ntasks = 10;
sum_species = [30,10,60];

% Create species
s1 = ones(sum_species(1),1) * 1;
s2 = ones(sum_species(2),1) * 2;
s3 = ones(sum_species(3),1) * 3;
boats_species = [s1;s2;s3];

% Give species to each boat.
%boats_species = randi(nspecies, nboats, 1)
% Places N boats randomly (within 3x3 arena).
boats_init_pos = rand(nboats, 2) * arena_size;

%% Assign tasks to boats.
boats_init_task = randi(ntasks, nboats, 1);

init_s1 = [ 10.45096636,   2.48212032,  14.37125749;...
             3.57909807,   3.53386622,  13.87569688;...
            10.02147459,   2.48212032,  20.56576502;...
             1.39584825,   0.22717711,   0.4088375;...
             0.64423765,   0.0252419 ,   2.15568862;...
             0.10737294,   0.46697518,   2.89902953;...
             0.70866142,   0.22717711,   1.78401817;...
             1.00930565,   0.12620951,   0.63183977;...
             0.68718683,   0.05679428,   3.08486475;...
             1.39584825,   0.37231805,   0.22300227];

init_s2 = [  1.46191249e+00,   8.37758784e-02,   4.34002781e+00;...
             2.94420569e-03,   2.69414674e-01,   1.14143018e-02;...
             8.24377592e-03,   2.45507949e+00,   1.69176259e-02;...
             1.38660755e+01,   1.83859947e+00,   2.26560059e+01;...
             6.91066150e+00,   1.92498558e+00,   1.41000332e+01;...
             3.30981364e+00,   9.90877011e-02,   6.41569511e+00;...
             1.79232304e+00,   3.08138374e-04,   3.74903014e+00;...
             6.99318037e-01,   8.24059662e-01,   3.00172331e+00;...
             8.49478662e-01,   1.65555003e+00,   3.44724066e+00;...
             1.09922914e+00,   8.49139379e-01,   2.26191193e+00];

init_s3 = [  6.66732600e-01,   3.87571614e-01,   4.57503571e+00;...
             1.00488157e-05,   2.75942194e-01,   3.52752793e-01;...
             1.08051538e+00,   1.10628202e+00,   3.34452196e+00;...
             1.67459737e+00,   1.83859947e+00,   4.08971026e+00;...
             1.96417691e+00,   2.22528466e-03,   4.15907015e+00;...
             9.82023721e+00,   3.88090406e+00,   1.66246763e+01;...
             1.04655682e+01,   1.18570106e+00,   1.35649490e+01;...
             6.52085305e-01,   3.40517465e-01,   5.21833099e+00;...
             1.37487424e+00,   6.22477084e-01,   5.65456870e+00;...
             2.30120271e+00,   3.59779752e-01,   2.41638409e+00];

if auto_advance
    slot = 1
end
switch slot
    case 1
        init_sx = init_s1;
    case 2
        init_sx = init_s2;
    case 3
        init_sx = init_s3;
end

%%%% Modified HERE %%%%%
temp_i = 1;
max_temp_i = 0;
for si = 1:nspecies
    max_temp_i = max_temp_i + sum_species(si);
    for ti = 1:ntasks
        nb = round(init_sx(ti,si));
        for nbi = 1:nb
            boats_init_task(temp_i) = ti;
            temp_i = temp_i + 1;
            % Avoid eventual rounding problems.
            if temp_i > max_temp_i
                break;
            end
        end
    end
    % There may still be boats to distribute.
    % Pick randomly.
    while temp_i <= max_temp_i
        boats_init_task(temp_i) = randsample(1:ntasks, 1, true, init_sx(:,si));
        temp_i = temp_i + 1;
    end
end
%%%% END modified %%%%%

%% Trajectories of the boats.
boats_pos = zeros(nboats, length(T), 2);
boats_task = zeros(nboats, length(T), 1);
for i = 1:nboats
    boats_pos(i, 1, :) = boats_init_pos(i, :);
    boats_task(i, 1, :) = boats_init_task(i, :);
end


%% Initialize transition values

% Optimized K matrices (to be taken from the Python code).
% Rates must be small must be low (in the order of a boat make 1-2 turns around the task site).

fact = 0.01;
K = cell(nslots);

% t0 - t1
% slot==1
K{1} = cell(nspecies);
K{1}{1} = fact * [-1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0; ...
                  0.0 -1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 -1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 -0.411219553868 1.0 0.0 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 0.411219553868 -1.27680272657 1.0 0.0 0.0 0.0 0.0; ...
                  1.0 0.0 0.0 0.0 0.276802726571 -1.66430708329 1.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 0.0 0.0 0.664307083287 -1.72781742669 1.0 0.0 0.0; ...
                  0.0 0.0 1.0 0.0 0.0 0.0 0.727817426691 -2.0 0.0 0.0; ...
                  0.0 1.0 0.0 0.0 0.0 0.0 0.0 1.0 -1.0 0.0; ...
                  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 -1.0];
K{1}{2} = fact * [-1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0937223777343; ...
                  0.0 -0.952597766181 0.0 0.0 0.0 0.0 0.0 0.0 0.147921420225 0.0; ...
                  0.0 0.0 -0.329011506846 0.0 0.0 0.0 0.0 0.995614253644 0.0 0.0; ...
                  0.0 0.0 0.0 -0.673659050894 0.675307291121 0.0 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 0.673659050894 -0.675307291121 1.0 0.0 0.0 0.0 0.0; ...
                  1.0 0.0 0.0 0.0 0.0 -1.0 0.930922743984 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 0.0 0.0 0.0 -0.930922743984 0.0 0.0 0.0; ...
                  0.0 0.0 0.329011506846 0.0 0.0 0.0 0.0 -1.92336718369 0.464226539878 0.0; ...
                  0.0 0.952597766181 0.0 0.0 0.0 0.0 0.0 0.927752930043 -1.14687190245 0.97943099747; ...
                  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.534723942345 -1.0731533752];
K{1}{3} = fact * [-1.47791832777 0.0 0.0 0.0 0.0 0.930561990126 0.0 0.0 0.0 0.0; ...
                  0.0 -1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 -1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 -0.525711046125 1.0 0.0 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 0.525711046125 -1.25985616218 1.0 0.0 0.0 0.0 0.0; ...
                  1.0 0.0 0.0 0.0 0.259856162183 -1.93056199013 1.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 0.0 0.0 0.0 -1.0 1.0 0.0 0.0; ...
                  0.0 0.0 1.0 0.0 0.0 0.0 0.0 -1.0 0.745885969262 0.0; ...
                  0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.745885969262 1.0; ...
                  0.477918327774 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -1.0];

% t1 - t2
% slot==2
K{2} = cell(nspecies);
K{2}{1} = fact * [-1.0 0.0 0.0 0.0 0.0 0.0682470953191 0.0 0.0 0.0 0.0; ...
                  0.0 -0.802935004264 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 -0.510164647348 0.0 0.0 0.0 0.0 1.0 0.0 0.0; ...
                  0.0 0.0 0.0 -1.0 0.747547577241 0.0 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 1.0 -1.74754757724 0.163108861824 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 0.0 1.0 -1.23135595714 0.966819468177 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 0.0 0.0 1.0 -0.989520697783 1.0 0.0 0.0; ...
                  0.0 0.0 0.510164647348 0.0 0.0 0.0 0.0227012296059 -2.0 0.404350218333 0.0; ...
                  0.0 0.802935004264 0.0 0.0 0.0 0.0 0.0 0.0 -0.404350218333 0.264912853753; ...
                  1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.264912853753];
K{2}{2} = fact * [-1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0; ...
                  0.0 -0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.00105034155021 0.0; ...
                  0.0 0.0 -0.346080021259 0.0 0.0 0.0 0.0 1.0 0.0 0.0; ...
                  0.0 0.0 0.0 -0.0 0.0 0.0 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 0.0 -0.953024689036 0.0 0.0 0.0 0.0 0.0; ...
                  1.0 0.0 0.0 0.0 0.953024689036 -0.0890780679719 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 0.0 0.0 0.0890780679719 -0.541520125069 1.0 0.0 0.0; ...
                  0.0 0.0 0.346080021259 0.0 0.0 0.0 0.541520125069 -3.0 0.0 0.0; ...
                  0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 -0.558206682952 0.0; ...
                  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.557156341401 -1.0];
K{2}{3} = fact * [-1.0 0.0 0.0 0.0 0.0 0.140860443126 0.0 0.0 0.0 1.0; ...
                  0.0 -1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0666103211621 0.0; ...
                  0.0 0.0 -1.0 0.0 0.0 0.0 0.0 0.665559705753 0.0 0.0; ...
                  0.0 0.0 0.0 -1.0 0.879274810879 0.0 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 1.0 -1.87927481088 0.205825305236 0.0 0.0 0.0 0.0; ...
                  1.0 0.0 0.0 0.0 1.0 -1.34668574836 0.997907396149 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 0.0 0.0 1.0 -1.60967831094 1.0 0.0 0.0; ...
                  0.0 0.0 1.0 0.0 0.0 0.0 0.611770914794 -2.66555970575 0.419839219279 0.0; ...
                  0.0 1.0 0.0 0.0 0.0 0.0 0.0 1.0 -0.942396210217 0.0; ...
                  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.455946669776 -1.0];

% t2 - t3
% slot==3
K{3} = cell(nspecies);
K{3}{1} = fact * [-1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0; ...
                  0.0 -0.958618658436 0.0 0.0 0.0 0.0 0.0 0.0 0.0049582057256 0.0; ...
                  0.0 0.0 -0.133356666343 0.0 0.0 0.0 0.0 0.100651012704 0.0 0.0; ...
                  0.0 0.0 0.0 -3.75907247724e-11 0.814887531129 0.0 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 3.75907247724e-11 -1.81488753113 4.55101736159e-11 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 0.0 1.0 -1.53776448819 0.945681956897 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 0.0 0.0 0.537764488148 -1.9456819569 0.694062947027 0.0 0.0; ...
                  0.0 0.0 0.133356666343 0.0 0.0 0.0 1.0 -1.79471395973 0.368842882419 0.0; ...
                  0.0 0.958618658436 0.0 0.0 0.0 0.0 0.0 1.0 -1.34398562755 0.663936948658; ...
                  1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.970184539402 -0.663936948658];
K{3}{2} = fact * [-1.53640138705 0.0 0.0 0.0 0.0 0.930339953774 0.0 0.0 0.0 0.0; ...
                  0.0 -0.887186278623 0.0 0.0 0.0 0.0 0.0 0.0 0.00200235388688 0.0; ...
                  0.0 0.0 -0.945942156884 0.0 0.0 0.0 0.0 0.211634405745 0.0 0.0; ...
                  0.0 0.0 0.0 -1.0 0.0 0.0 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 1.0 -0.000536282397598 0.00259968514327 0.0 0.0 0.0 0.0; ...
                  0.536401387049 0.0 0.0 0.0 0.000536282397598 -1.33844149696 1.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 0.0 0.0 0.405501858046 -2.0 0.326053399986 0.0 0.0; ...
                  0.0 0.0 0.945942156884 0.0 0.0 0.0 1.0 -1.53761782169 0.532405021726 0.0; ...
                  0.0 0.887186278623 0.0 0.0 0.0 0.0 0.0 0.999930015954 -1.10686085324 1.0; ...
                  1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.572453477626 -1.0];
K{3}{3} = fact * [-1.69994279409 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.445315587323; ...
                  0.0 -0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.00180801135563 0.0; ...
                  0.0 0.0 -0.817853836128 0.0 0.0 0.0 0.0 0.097748075425 0.0 0.0; ...
                  0.0 0.0 0.0 -0.0115115380078 0.0 0.0 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 0.0115115380078 -1.0 0.00490722785904 0.0 0.0 0.0 0.0; ...
                  0.69994279409 0.0 0.0 0.0 1.0 -2.00490722786 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 0.0 0.0 1.0 -1.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.817853836128 0.0 0.0 0.0 1.0 -1.09774807543 0.798103278872 0.0; ...
                  0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 -1.2000477979 1.0; ...
                  1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.40013650767 -1.44531558732];

P = cell(nslots);
for j = 1:nslots
    P{j} = cell(nspecies);
    for i = 1:nspecies
        P{j}{i} = expm(K{j}{i} * dt);
    end
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
current_P = P{slot};
switching_steps = zeros(1, nslots + 1);
switching_steps(1) = 1;
switching_steps(end) = size(T, 2) + 1;
i = 1;
while i < size(T, 2)
    i = i + 1;
    if(mod(i,50)==0)
        txt = sprintf('%d / %d',i,size(T, 2));
        disp(txt);
    end
    t = T(i);
    if auto_advance && t > setup_time && slot < nslots && mod(i - dt_for_setup - 1, dt_per_slot) == 0
        t
        slot = slot + 1
        switching_steps(slot) = i;
        current_P = P{slot};
    end
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
                ntask = randsample(1:ntasks, 1, true, current_P{boats_species(j)}(:, task));
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

if plot_on
    figure_handle = figure(1);
    plot_arena(figure_handle, task_sites, boats_pos, squeeze(boats_task(:,end,:)), make_movie);
end
