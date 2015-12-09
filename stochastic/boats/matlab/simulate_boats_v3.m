% Simple code that runs boats in simulation
% Some values are hardcoded:
% species_traits
% sum_species (#robots per species)


%% Setup

clear
file_time = datestr(clock);

tic

run = 510;

verbose = false;

% from the trait evolution data, choose which time slot [1,2,3]
slot = 1;
% If true, the simulation will cycle through the different slots.
nslots = 3;
auto_advance = true;
boat_multiplier = 5; % total is multiple of 100

save_data = true;
ws_filename = strcat('./data/run_',int2str(run),'_all_data.mat'); %strcat(file_time,'_data.mat');
vars_filename = strcat('./data/run_',int2str(run),'_vars_data.mat'); %strcat(file_time,'_pos.mat');

make_movie = false;
show_plot = false;

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
nboats = 100 * boat_multiplier;
nspecies = 3;
ntasks = 10;
sum_species = [30,10,60] * boat_multiplier;

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
             1.39584825,   0.37231805,   0.22300227] * boat_multiplier;

init_s2 = [  1.46191249e+00,   8.37758784e-02,   4.34002781e+00;...
             2.94420569e-03,   2.69414674e-01,   1.14143018e-02;...
             8.24377592e-03,   2.45507949e+00,   1.69176259e-02;...
             1.38660755e+01,   1.83859947e+00,   2.26560059e+01;...
             6.91066150e+00,   1.92498558e+00,   1.41000332e+01;...
             3.30981364e+00,   9.90877011e-02,   6.41569511e+00;...
             1.79232304e+00,   3.08138374e-04,   3.74903014e+00;...
             6.99318037e-01,   8.24059662e-01,   3.00172331e+00;...
             8.49478662e-01,   1.65555003e+00,   3.44724066e+00;...
             1.09922914e+00,   8.49139379e-01,   2.26191193e+00] * boat_multiplier;

init_s3 = [  6.66732600e-01,   3.87571614e-01,   4.57503571e+00;...
             1.00488157e-05,   2.75942194e-01,   3.52752793e-01;...
             1.08051538e+00,   1.10628202e+00,   3.34452196e+00;...
             1.67459737e+00,   1.83859947e+00,   4.08971026e+00;...
             1.96417691e+00,   2.22528466e-03,   4.15907015e+00;...
             9.82023721e+00,   3.88090406e+00,   1.66246763e+01;...
             1.04655682e+01,   1.18570106e+00,   1.35649490e+01;...
             6.52085305e-01,   3.40517465e-01,   5.21833099e+00;...
             1.37487424e+00,   6.22477084e-01,   5.65456870e+00;...
             2.30120271e+00,   3.59779752e-01,   2.41638409e+00] * boat_multiplier;

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
K{1} = cell(nspecies);
K{1}{1} = fact * [-2.34290352215 0.0 0.0 0.0 0.0 0.402781726329 0.0 0.0 0.0 2.0; ...
                  0.0 -2.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 -2.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 -0.988112390718 2.0 0.0 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 0.988112390718 -2.93415152865 2.0 0.0 0.0 0.0 0.0; ...
                  2.0 0.0 0.0 0.0 0.934151528652 -3.8397991491 2.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 0.0 0.0 1.43701742277 -3.60274261228 2.0 0.0 0.0; ...
                  0.0 0.0 2.0 0.0 0.0 0.0 1.60274261228 -4.0 0.0 0.0; ...
                  0.0 2.0 0.0 0.0 0.0 0.0 0.0 2.0 -2.0 0.0; ...
                  0.342903522151 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.0 -2.0];
K{1}{2} = fact * [-1.99978483944 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0434240857088; ...
                  0.0 -1.99997016975 0.0 0.0 0.0 0.0 0.0 0.0 0.314481068779 0.0; ...
                  0.0 0.0 -0.574755909471 0.0 0.0 0.0 0.0 1.57067761588 0.0 0.0; ...
                  0.0 0.0 0.0 -1.42707927093 1.43944341979 0.0 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 1.42707927093 -1.43944341979 2.0 0.0 0.0 0.0 0.0; ...
                  1.99978483944 0.0 0.0 0.0 0.0 -2.0 2.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 0.0 0.0 0.0 -2.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.574755909471 0.0 0.0 0.0 0.0 -2.98701497904 0.708408728206 0.0; ...
                  0.0 1.99997016975 0.0 0.0 0.0 0.0 0.0 1.41633736316 -1.83805906042 1.72013739029; ...
                  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.815169263435 -1.763561476];
K{1}{3} = fact * [-3.98365274416 0.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0 1.6539524183; ...
                  0.0 -2.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 -2.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 -1.23067993291 2.0 0.0 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 1.23067993291 -2.89093485334 2.0 0.0 0.0 0.0 0.0; ...
                  1.99852102117 0.0 0.0 0.0 0.890934853344 -4.34481595622 2.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 0.0 0.0 0.344815956219 -2.19072504445 2.0 0.0 0.0; ...
                  0.0 0.0 2.0 0.0 0.0 0.0 0.190725044445 -2.0 1.47876052034 0.0; ...
                  0.0 2.0 0.0 0.0 0.0 0.0 0.0 0.0 -1.47876052034 2.0; ...
                  1.98513172299 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -3.6539524183];

K{2} = cell(nspecies);
K{2}{1} = fact * [-2.0 0.0 0.0 0.0 0.0 0.136624515096 0.0 0.0 0.0 0.0; ...
                  0.0 -1.97038161074 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 -1.26468161514 0.0 0.0 0.0 0.0 2.0 0.0 0.0; ...
                  0.0 0.0 0.0 -2.0 1.54007420326 0.0 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 2.0 -3.54007420326 0.379004497993 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 0.0 2.0 -2.08119514102 1.56451308466 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 0.0 0.0 1.56556612793 -1.59251538912 2.0 0.0 0.0; ...
                  0.0 0.0 1.26468161514 0.0 0.0 0.0 0.0280023044616 -4.0 0.922337982699 0.0; ...
                  0.0 1.97038161074 0.0 0.0 0.0 0.0 0.0 0.0 -0.922337982699 0.581214271256; ...
                  2.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.581214271256];
K{2}{2} = fact * [-2.0 0.0 0.0 0.0 0.0 0.220357576537 0.0 0.0 0.0 0.0; ...
                  0.0 -2.0 0.0 0.0 0.0 0.0 0.0 0.0 0.852529488926 0.0; ...
                  0.0 0.0 -0.92309123903 0.0 0.0 0.0 0.0 2.0 0.0 0.0; ...
                  0.0 0.0 0.0 -0.0 0.0354240931866 0.0 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 0.0 -1.10838499758 0.0 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 0.0 1.0729609044 -0.360556226234 1.31156424235 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 0.0 0.0 0.140198649697 -1.31156424235 2.0 0.0 0.0; ...
                  0.0 0.0 0.92309123903 0.0 0.0 0.0 0.0 -4.62903549268 1.66275495851 0.0; ...
                  0.0 2.0 0.0 0.0 0.0 0.0 0.0 0.629035492678 -2.51528444743 2.0; ...
                  2.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -2.0];
K{2}{3} = fact * [-2.0 0.0 0.0 0.0 0.0 0.266732796483 0.0 0.0 0.0 2.0; ...
                  0.0 -2.0 0.0 0.0 0.0 0.0 0.0 0.0 0.123515839547 0.0; ...
                  0.0 0.0 -2.0 0.0 0.0 0.0 0.0 1.30902428136 0.0 0.0; ...
                  0.0 0.0 0.0 -2.0 1.80047077744 0.0 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 2.0 -3.80047077744 0.472013172614 0.0 0.0 0.0 0.0; ...
                  2.0 0.0 0.0 0.0 2.0 -2.69503513844 2.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 0.0 0.0 1.95628916934 -3.15545535192 2.0 0.0 0.0; ...
                  0.0 0.0 2.0 0.0 0.0 0.0 1.15545535192 -5.30902428136 0.914085965797 0.0; ...
                  0.0 2.0 0.0 0.0 0.0 0.0 0.0 2.0 -1.92490059378 0.0; ...
                  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.887298788441 -2.0];

K{3} = cell(nspecies);
K{3}{1} = fact * [-2.0 0.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0 0.0; ...
                  0.0 -1.99598230584 0.0 0.0 0.0 0.0 0.0 0.0 0.00818842055577 0.0; ...
                  0.0 0.0 -1.45256014233 0.0 0.0 0.0 0.0 0.933443859256 0.0 0.0; ...
                  0.0 0.0 0.0 -0.0 1.78344463678 0.0 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 0.0 -3.68796300293 0.0526050140491 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 0.0 1.90451836615 -3.04208267168 1.72019181219 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 0.0 0.0 0.989477657635 -3.72019181219 1.25325850962 0.0 0.0; ...
                  0.0 0.0 1.45256014233 0.0 0.0 0.0 2.0 -4.18670236888 0.715484656674 0.0; ...
                  0.0 1.99598230584 0.0 0.0 0.0 0.0 0.0 2.0 -2.54471145749 1.24532463923; ...
                  2.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.82103838026 -1.24532463923];
K{3}{2} = fact * [-2.51572463307 0.0 0.0 0.0 0.0 1.58259158635 0.0 0.0 0.0 0.0; ...
                  0.0 -1.70203130685 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 -1.65466560764 0.0 0.0 0.0 0.0 0.364338922209 0.0 0.0; ...
                  0.0 0.0 0.0 -1.99981087449 0.00138559910843 0.0 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 1.99981087449 -0.00138559910843 4.83161941029e-05 0.0 0.0 0.0 0.0; ...
                  0.515724633071 0.0 0.0 0.0 0.0 -2.42546806975 1.99219970993 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 0.0 0.0 0.842828167207 -3.78667122532 0.663799394777 0.0 0.0; ...
                  0.0 0.0 1.65466560764 0.0 0.0 0.0 1.79447151539 -2.93223743061 1.00261516106 0.0; ...
                  0.0 1.70203130685 0.0 0.0 0.0 0.0 0.0 1.90409911362 -2.09253285006 1.96772361043; ...
                  2.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.089917689 -1.96772361043];
K{3}{3} = fact * [-3.3151708597 0.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0 0.86194417465; ...
                  0.0 -1.93690624317 0.0 0.0 0.0 0.0 0.0 0.0 0.0525527924656 0.0; ...
                  0.0 0.0 -1.35985666219 0.0 0.0 0.0 0.0 0.164078672589 0.0 0.0; ...
                  0.0 0.0 0.0 -1.06304577781e-11 0.0918291285905 0.0 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 1.06304577781e-11 -2.09182912859 0.0 0.0 0.0 0.0 0.0; ...
                  1.3151708597 0.0 0.0 0.0 2.0 -3.83320637583 0.0 0.0 0.0 0.0; ...
                  0.0 0.0 0.0 0.0 0.0 1.83320637583 -2.0 0.0 0.0 0.0; ...
                  0.0 0.0 1.35985666219 0.0 0.0 0.0 2.0 -2.16376886137 1.59924360977 0.0; ...
                  0.0 1.93690624317 0.0 0.0 0.0 0.0 0.0 1.99969018878 -2.44360785682 1.99670969925; ...
                  2.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.791811454579 -2.8586538739];

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
                if verbose
                    temp = sprintf('Boat %d switched to task %d\n', j, ntask);
                    disp(temp)
                end
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
    %save(vars_filename,'boats_pos','task_sites','boats_task','T', 'dt', 'boats_species');
end

if show_plot
    figure_handle = figure(1);
    plot_arena(figure_handle, task_sites, boats_pos, squeeze(boats_task(:,end,:)), make_movie);
end

toc
