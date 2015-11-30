% Simple code that runs boats in simulation
% Some values are hardcoded:
% species_traits
% sum_species (#robots per species)


%% Setup

clear
file_time = datestr(clock);

run = 100;

% from the trait evolution data, choose which time slot [1,2,3]
slot = 1;
% If true, the simulation will cycle through the different slots.
nslots = 3;
auto_advance = true;

save_data = true;
ws_filename = strcat('run_',int2str(run),'_all_data.mat'); %strcat(file_time,'_data.mat');
vars_filename = strcat('run_',int2str(run),'_vars_data.mat'); %strcat(file_time,'_pos.mat');
verbose = false

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

% Ks created for tmax=7s, simulation of boats is 700s
fact = 0.01;
K = cell(nslots);

% t0 - t1
% slot==1
K{1} = cell(nspecies);
K{1}{1} = fact * [-1.1994028883 0.0 0.0 0.0 0.0 0.554456888468 0.0 0.0 0.0 1.0; ...
        0.0 -1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; ...
        0.0 0.0 -1.0 0.0 0.0 0.0 0.0 2.33203571082e-12 0.0 0.0; ...
        0.0 0.0 0.0 -0.897622519918 0.393327850702 0.0 0.0 0.0 0.0 0.0; ...
        0.0 0.0 0.0 0.897622519918 -0.518816909213 1.0 0.0 0.0 0.0 0.0; ...
        1.0 0.0 0.0 0.0 0.125489058511 -1.68625037357 1.0 0.0 0.0 0.0; ...
        0.0 0.0 0.0 0.0 0.0 0.131793485101 -1.77819701498 1.0 0.0 0.0; ...
        0.0 0.0 1.0 0.0 0.0 0.0 0.778197014984 -1.01622716127 1.0 0.0; ...
        0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0162271612683 -2.0 0.847200648183; ...
        0.199402888295 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 -1.84720064818];

K{1}{2} = fact * [-1.0 0.0 0.0 0.0 0.0 2.33203571082e-12 0.0 0.0 0.0 0.973821811229; ...
        0.0 -0.945435961265 0.0 0.0 0.0 0.0 0.0 0.0 0.352240778123 0.0; ...
        0.0 0.0 -0.18252625516 0.0 0.0 0.0 0.0 0.77950487639 0.0 0.0; ...
        0.0 0.0 0.0 -0.407660316809 0.459915595311 0.0 0.0 0.0 0.0 0.0; ...
        0.0 0.0 0.0 0.407660316809 -0.459915595311 0.999999999998 0.0 0.0 0.0 0.0; ...
        1.0 0.0 0.0 0.0 0.0 -1.0 1.0 0.0 0.0 0.0; ...
        0.0 0.0 0.0 0.0 0.0 2.33203571082e-12 -1.37990261737 0.00196664428095 0.0 0.0; ...
        0.0 0.0 0.18252625516 0.0 0.0 0.0 0.379902617366 -1.53372449716 0.322704059282 0.0; ...
        0.0 0.945435961265 0.0 0.0 0.0 0.0 0.0 0.752252976494 -0.675661283448 0.451444790409; ...
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.000716446042495 -1.42526660164];

K{1}{3} = fact * [-1.78107244815 0.0 0.0 0.0 0.0 0.471851714506 0.0 0.0 0.0 0.107428360871; ...
        0.0 -1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.956846698414 0.0; ...
        0.0 0.0 -1.0 0.0 0.0 0.0 0.0 0.501339272315 0.0 0.0; ...
        0.0 0.0 0.0 -0.213400572017 0.0428409173404 0.0 0.0 0.0 0.0 0.0; ...
        0.0 0.0 0.0 0.213400572017 -0.212541168663 1.0 0.0 0.0 0.0 0.0; ...
        1.0 0.0 0.0 0.0 0.169700251323 -1.47185171451 1.0 0.0 0.0 0.0; ...
        0.0 0.0 0.0 0.0 0.0 0.0 -1.0 1.0 0.0 0.0; ...
        0.0 0.0 1.0 0.0 0.0 0.0 0.0 -1.50133927231 1.0 0.0; ...
        0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 -1.95684669841 0.83748286431; ...
        0.781072448148 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.944911225181];


% t1 - t2
% slot==2
K{2} = cell(nspecies);
K{2}{1} = fact *  [-1.0 0.0 0.0 0.0 0.0 0.131794922989 0.0 0.0 0.0 1.0; ...
        0.0 -0.730620318934 0.0 0.0 0.0 0.0 0.0 0.0 0.0549204406422 0.0; ...
        0.0 0.0 -0.853385003893 0.0 0.0 0.0 0.0 0.337119837522 0.0 0.0; ...
        0.0 0.0 0.0 -0.999672121247 0.386544226329 0.0 0.0 0.0 0.0 0.0; ...
        0.0 0.0 0.0 0.999672121247 -1.38654422633 0.263503331803 0.0 0.0 0.0 0.0; ...
        1.0 0.0 0.0 0.0 1.0 -0.531216228433 0.987448893431 0.0 0.0 0.0; ...
        0.0 0.0 0.0 0.0 0.0 0.135917973641 -1.98744889343 0.0119707252251 0.0 0.0; ...
        0.0 0.0 0.853385003893 0.0 0.0 0.0 1.0 -1.27178940034 0.34879293745 0.0; ...
        0.0 0.730620318934 0.0 0.0 0.0 0.0 0.0 0.922698837589 -1.24917583863 0.965812164924; ...
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.845462460542 -1.96581216492];

K{2}{2} = fact * [-1.35941259092 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.882572649506; ...
        0.0 -0.956164357371 0.0 0.0 0.0 0.0 0.0 0.0 0.227582308207 0.0; ...
        0.0 0.0 -0.39775781436 0.0 0.0 0.0 0.0 0.484537630283 0.0 0.0; ...
        0.0 0.0 0.0 -1.0 0.0 0.0 0.0 0.0 0.0 0.0; ...
        0.0 0.0 0.0 1.0 -0.99325916737 0.0 0.0 0.0 0.0 0.0; ...
        0.89236281274 0.0 0.0 0.0 0.99325916737 -0.817001613258 0.893742531892 0.0 0.0 0.0; ...
        0.0 0.0 0.0 0.0 0.0 0.817001613258 -1.24571769234 0.491672303124 0.0 0.0; ...
        0.0 0.0 0.39775781436 0.0 0.0 0.0 0.351975160443 -1.51703078612 0.577306362879 0.0; ...
        0.0 0.956164357371 0.0 0.0 0.0 0.0 0.0 0.540820852716 -0.886119042188 0.836417854318; ...
        0.467049778179 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0812303711019 -1.71899050382];

K{2}{3} = fact *  [-1.42303814745 0.0 0.0 0.0 0.0 0.0487342807736 0.0 0.0 0.0 0.138911341262; ...
        0.0 -0.62203401799 0.0 0.0 0.0 0.0 0.0 0.0 0.760901483486 0.0; ...
        0.0 0.0 -0.24462648579 0.0 0.0 0.0 0.0 1.0 0.0 0.0; ...
        0.0 0.0 0.0 -1.0 0.134086204673 0.0 0.0 0.0 0.0 0.0; ...
        0.0 0.0 0.0 1.0 -1.13408620467 0.142290549694 0.0 0.0 0.0 0.0; ...
        0.960770863188 0.0 0.0 0.0 1.0 -0.30096377597 0.964003309245 0.0 0.0 0.0; ...
        0.0 0.0 0.0 0.0 0.0 0.109938945502 -0.964003309245 0.265882959771 0.0 0.0; ...
        0.0 0.0 0.24462648579 0.0 0.0 0.0 0.0 -1.86171767419 0.47076336328 0.0; ...
        0.0 0.62203401799 0.0 0.0 0.0 0.0 0.0 0.595834714417 -1.66290434554 0.850768835468; ...
        0.462267284258 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.431239498777 -0.98968017673];

% t2 - t3
% slot==3
K{3} = cell(nspecies);
K{3}{1} = fact * [-1.0 0.0 0.0 0.0 0.0 0.0354586965629 0.0 0.0 0.0 1.0; ...
        0.0 -7.0276189992e-13 0.0 0.0 0.0 0.0 0.0 0.0 0.00491157190605 0.0; ...
        0.0 0.0 -3.12047611376e-09 0.0 0.0 0.0 0.0 0.00511171170184 0.0 0.0; ...
        0.0 0.0 0.0 -0.356132877867 0.193731893785 0.0 0.0 0.0 0.0 0.0; ...
        0.0 0.0 0.0 0.356132877867 -0.24970124775 0.0 0.0 0.0 0.0 0.0; ...
        0.0 0.0 0.0 0.0 0.0559693539651 -1.03545869656 0.0 0.0 0.0 0.0; ...
        0.0 0.0 0.0 0.0 0.0 1.0 -1.0 0.0 0.0 0.0; ...
        0.0 0.0 3.12047611376e-09 0.0 0.0 0.0 1.0 -0.991313402648 1.0 0.0; ...
        0.0 7.0276189992e-13 0.0 0.0 0.0 0.0 0.0 0.986201690946 -1.09006961725 1.0; ...
        1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0851580453474 -2.0];

K{3}{2} = fact * [-1.99368886856 0.0 0.0 0.0 0.0 0.0257280118782 0.0 0.0 0.0 0.0; ...
        0.0 -0.898450365059 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; ...
        0.0 0.0 -1.0 0.0 0.0 0.0 0.0 0.128362740868 0.0 0.0; ...
        0.0 0.0 0.0 -0.556606598054 0.0902496357029 0.0 0.0 0.0 0.0 0.0; ...
        0.0 0.0 0.0 0.556606598054 -0.532559730663 0.0 0.0 0.0 0.0 0.0; ...
        0.993688868557 0.0 0.0 0.0 0.442310094961 -0.0257280118782 0.0 0.0 0.0 0.0; ...
        0.0 0.0 0.0 0.0 0.0 0.0 -1.0 0.0 0.0 0.0; ...
        0.0 0.0 1.0 0.0 0.0 0.0 1.0 -0.992060532944 0.798648015981 0.0; ...
        0.0 0.898450365059 0.0 0.0 0.0 0.0 0.0 0.863697792076 -0.879169762141 0.635876943924; ...
        1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0805217461602 -0.635876943924];

K{3}{3} = fact * [-2.0 0.0 0.0 0.0 0.0 0.588387983146 0.0 0.0 0.0 0.319141872723; ...
        0.0 -0.0809846849394 0.0 0.0 0.0 0.0 0.0 0.0 0.013848655064 0.0; ...
        0.0 0.0 -0.304003233632 0.0 0.0 0.0 0.0 0.148961813488 0.0 0.0; ...
        0.0 0.0 0.0 -0.55334281871 0.0996181152103 0.0 0.0 0.0 0.0 0.0; ...
        0.0 0.0 0.0 0.55334281871 -0.100968374847 0.00679864499167 0.0 0.0 0.0 0.0; ...
        1.0 0.0 0.0 0.0 0.00135025963679 -1.24891418052 0.24267909403 0.0 0.0 0.0; ...
        0.0 0.0 0.0 0.0 0.0 0.65372755238 -1.24267909403 0.0 0.0 0.0; ...
        0.0 0.0 0.304003233632 0.0 0.0 0.0 1.0 -1.14896181349 0.468972551152 0.0; ...
        0.0 0.0809846849394 0.0 0.0 0.0 0.0 0.0 1.0 -1.10954428593 1.0; ...
        1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.626723079719 -1.31914187272];


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
    save(vars_filename,'boats_pos','task_sites','boats_task','T', 'dt', 'boats_species');
end

if plot_on
    figure_handle = figure(1);
    plot_arena(figure_handle, task_sites, boats_pos, squeeze(boats_task(:,end,:)), make_movie);
end
