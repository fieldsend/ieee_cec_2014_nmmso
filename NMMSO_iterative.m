function [mode_loc,mode_y,evaluations,nmmso_state] =  NMMSO_iterative( ...
    swarm_size, problem_func,problem_function_params, max_evaluations, ...
    mn,mx,evaluations,nmmso_state,max_evol,tol_val)

% Implementation of the Niching Migratory Multi-Swarm Optimser, described
% in:
% "Running Up Those Hills: Multi-Modal Search with the Niching Migratory 
% Multi-Swarm Optimiser"
% by Jonathan E. Fieldsend
% published in Proceedings of the IEEE Congress on Evolutionary Computation, 
% pages 2593-2600, 2014
%
% Please reference this paper if you undertake work utilising this code.
% Implementation (c) by Jonathan Fieldsend, University of Exeter, 2014
%
% Assumes function maximisation
%
% REQUIRED ARGUMENTS
%
% swarm_size = maximum number of elements (particles) per swarm
% problem_func = string containing name of function to be optimised
% problem_funcion_params = meta-parameters needed by problem function 
%   (distinct from optimisation (design) parameters
% max_evaluations = maximum number of evaluations to be taken through the
%   problem function
% mn = minimum design parameter values (a vector with param_num elements)
% mx = maximum design parameter values (a vector with param_num elements)
% evaluations = number of evaluations expended already. If set at 0, this
%   will initialise the swarm_state structure and run the algorithm for a
%   single generation, otherwise it will run it for a single generation
%   from the evaluations number inputted
% swarm_state = structure holding state of swarm. Can be empty or omitted 
%   if evaluations set at zero, must be provided if evals > 0
%
% OPTIONAL ARGUMENTS
%
% max_evol = maximum number of swarms to update in a generation. If not
%   provided this is set at 100
% tol_val = tolerance value for merging automatically (default 10^-6)
%
% OUTPUTS
%
% mode_loc = design space location of current mode estimates (swarm
%   gbests), note that at least one is likely to be very poor due to the
%   new swarm spawning at the end of each generation, and that these will
%   be a combination of both global and local mode estimate
% mode_y = function evalutions corresponding to the mode estimates
% evaluations = number of problem function evaluations until this point
% nmmso_state = structure holding the state of the swarms. Unless you want
%   to pick apart the details of how the algorithm searchs the space, then 
%   the only two elements you will probably be interested in are X and Y
%   which are preallocated matrices to hold all locations visited 
%   (therefore nmmso_state.X(1:evaluations,:) will hold all the design
%   space locations visited by the optimiser thus far.

if (evaluations<0)
   error('Must run algorithm for a positive number of function evaluations'); 
end
if exist('max_evol','var')==0
    display('default max_eval used, set at 100');
    max_evol=100;
end

if max_evol<=0
    display('Max_eval cannot be negative or zero, default max_eval used, set at 100');
    max_evol=100;
end

if exist('tol_val','var') ==0
    tol_val = 10^-6;
end


if evaluations==0
    % preallocate matrices for speed, with a buffer at end in case of final
    % generation exploring begound max_evaluations limit
    nmmso_state.X = zeros(max_evaluations+500,length(mx));
    nmmso_state.Y = zeros(max_evaluations+500,1);
    nmmso_state.index = 1;
    nmmso_state.converged_modes = 0;
    
    
    % get initial locations
    [nmmso_state] = get_initial_locations(nmmso_state,mn,mx);
    [swarm,nmmso_state] = evaluate_first(...
        nmmso_state.active_modes(1).swarm,problem_func,problem_function_params,nmmso_state,swarm_size,mn,mx);
    nmmso_state.active_modes(1).swarm=swarm;
    % track number of evaluations taken
    evaluations = 1;
    
    % keep modes in matrices for effciency on some computations
    nmmso_state.M_loc =  nmmso_state.active_modes(1).swarm.mode_location;
    nmmso_state.V_loc = nmmso_state.active_modes(1).swarm.mode_value;
    nmmso_state.tol_val = tol_val;
end

if (evaluations<max_evaluations) % if limited evalutions not already exhausted/exceeded
    % first see if modes should be merged together
    number_of_mid_evals=0;
    while sum(nmmso_state.active_modes_changed)>0
        [nmmso_state,merge_evals] = merge_swarms(nmmso_state,problem_func,problem_function_params,mn,mx);
        number_of_mid_evals = number_of_mid_evals+merge_evals; % track function evals used
    end
    
    % Now increment the swarms
    % if we have more than max_evol, then only increment a subset
    limit = min(max_evol,length(nmmso_state.active_modes));
    if limit>max_evol % have to select a subset
        if rand()<0.5 % select fittest
            [~, fit_I] = sort(nmmso_state.V_loc,'descend');
        else % select at random
            fit_I = randperm(length(nmmso_state.V_loc));
        end
    else
        fit_I = 1:limit; % can increment all
    end
    I2 = fit_I(1:limit);
    
    % increment
    for jj=1:length(I2)
        [nmmso_state] = increment_swarm(nmmso_state,I2(jj),mn,mx,swarm_size);
    end
    % evaluate new member/new locations of swarm member
    [nmmso_state,number_of_new_locations] = evaluate_new_locations(nmmso_state,problem_func,problem_function_params,I2);
    
    % attempt to split off a member from one of the swarms to seed a new
    % swarm (if detected to be on another peak)
    [nmmso_state,number_of_hive_samples] = hive(nmmso_state,problem_func,mn,mx,problem_function_params,max_evol,swarm_size);
    
    % create speculative new swarm, either at random in design space, or
    % via crossover
    if rand()<0.5 || (length(nmmso_state.active_modes)==1) || (length(mx)==1)
        number_of_evol_modes=0;
        [nmmso_state,number_rand_modes] = random_new(nmmso_state,problem_func,mn,mx,problem_function_params,swarm_size);
    else
        number_rand_modes=0;
        [nmmso_state,number_of_evol_modes] = evolve(nmmso_state,problem_func,mn,mx,problem_function_params,max_evol,swarm_size);
    end
    
    % update the total number of function evaluations used, with those
    % required at each of the algorithm stages
    evaluations = evaluations+number_of_mid_evals+number_of_new_locations+number_of_evol_modes+number_rand_modes+number_of_hive_samples;

    fprintf('Number of swarms %d, evals %d, max mode est. %f\n',length(nmmso_state.active_modes),evaluations,max(nmmso_state.V_loc));
    fprintf('\n');
else
    display('Evaluations taken already exhuasted/exceeded');
end

% get mode estimate locations
[mode_loc,mode_y] = extract_modes(nmmso_state);



%--------------------------------------------------------------------------
function [RES,RES_Y] = extract_modes(nmmso_state)

RES = zeros(length(nmmso_state.active_modes), length(nmmso_state.active_modes(1).swarm.mode_location));
RES_Y = zeros(length(nmmso_state.active_modes),1);

for i=1:length(nmmso_state.active_modes)
    RES(i,:) = nmmso_state.active_modes(i).swarm.mode_location;
    RES_Y(i) = nmmso_state.active_modes(i).swarm.mode_value;
end

%--------------------------------------------------------------------------
function [nmmso_state] = get_initial_locations(nmmso_state,mn,mx)

nmmso_state.active_modes(1).swarm.new_location=rand(size(mx)).*(mx-mn)+mn; 
nmmso_state.active_modes_changed(1) = 1;

%--------------------------------------------------------------------------
function [swarm,nmmso_state] = evaluate_first(swarm,problem_func,problem_function_params,nmmso_state,swarm_size,mn,mx)

% new_location is the only solution thus far in mode, so by definition is
% also the mode estimate, and the only history thus far

y = feval(problem_func, swarm.new_location,problem_function_params);
swarm.mode_location = swarm.new_location; %gbest location
swarm.mode_value = y; %gbest value


% initialise containers for swarm elements
swarm.history_locations = zeros(swarm_size,length(swarm.mode_location)); % current locations of swarm
swarm.history_values = ones(swarm_size,1)*-inf; % current values of swarm

swarm.pbest_locations = zeros(swarm_size,length(swarm.mode_location)); % current best locations of swarm
swarm.pbest_values = ones(swarm_size,1)*-inf; % current best locations of swarm

swarm.velocities = rand(size(mx)).*(mx-mn)+mn; % random initial velocities of swarm
swarm.number_of_particles = 1;

swarm.history_locations(1,:) = swarm.mode_location;
swarm.history_values(1) = y;

swarm.pbest_locations(1,:) = swarm.mode_location;
swarm.pbest_values(1) = y;

% track all made
nmmso_state.X(nmmso_state.index,:) = swarm.new_location;
nmmso_state.Y(nmmso_state.index)= y;
nmmso_state.index = nmmso_state.index +1;


%--------------------------------------------------------------------------
function [nmmso_state,number_of_mid_evals] = merge_swarms(nmmso_state,problem_func,problem_function_params,mn,mx)


% only concern ourselves with modes that have actually shifted, or are new
% since the last generation, as no need to check others
I = find(nmmso_state.active_modes_changed==1);
nmmso_state.active_modes_changed = nmmso_state.active_modes_changed*0; % reset 

n = length(I);
number_of_mid_evals=0;
if n>=1 && (length(nmmso_state.active_modes)>1) % only compare if there is a changed mode, and more than one mode in system
    to_compare = zeros(n,2);
    to_compare(:,1) = I;
    for i=1:n
        d = dist2(nmmso_state.M_loc(I(i),:),nmmso_state.M_loc); % dist2 can be found in the freely available netlab tool suite
        d(I(i)) = inf; % will be closest to itself, so need to get second closest
        [tmp, to_compare(i,2)] = min(d);
        nmmso_state.active_modes(I(i)).swarm.dist = sqrt(tmp); % track Euc dist to nearest neighbour mode
        
        if nmmso_state.active_modes(I(i)).swarm.number_of_particles==1
            reject=0;
            % in situation where a new swarm, and therefore distance to  
            % neighbour swarm now calculated
            % so set the initial velocity at a more reasonable value for
            % the first particle, rather than using the uniform in design space 
            temp_vel = mn-1;
            while (sum(temp_vel < mn)> 0 || sum(temp_vel > mx)>0)
                temp_vel = uniform_sphere_points(1,length(nmmso_state.active_modes(I(i)).swarm.new_location))*(nmmso_state.active_modes(I(i)).swarm.dist/2);
                reject = reject+1;
                if reject>20
                    % rejecting lots, so likely in a corner of design space
                    % where a significant volume of the sphere lies outside
                    % the bounds, so will make do with a random legal
                    % velocity in bounds
                    temp_vel = rand(size(nmmso_state.active_modes(I(i)).swarm.new_location)).*(mx-mn) + mn;
                end
            end
            nmmso_state.active_modes(I(i)).swarm.velocities(1,:) = temp_vel;
        end
    end
    
    % to_compare now contains the pairs of indices of closest modes, where at
    % least one mode has shifted location/is new since last generation. However,
    % there may be duplicated pairs (through reversals), so need to omit these.
    to_compare = sort(to_compare,2);
    % now sorted so that first column elements are always smaller than second
    % column elemets on same row
    [~,ind]=sort(to_compare(:,1));
    to_compare = to_compare(ind,:); %now sorted so that first column is sorted smallest to highest
    
    % remove_duplicates
    for i=n:-1:2
        I = find(to_compare(:,1) == to_compare(i,1)); %get indices of all with first index element same
        if sum(sum(repmat(to_compare(i,:),length(I),1)==to_compare(I,:),2)==2)>1 % if more than one vector duplication
            to_compare(i,:)=[];
        end
    end
    
    % now check for merging
    n = size(to_compare,1);
    to_merge =[];
    number_of_mid_evals=0;
    
    for i=1:n
        % merge if sufficiently close
        if sqrt(dist2(nmmso_state.active_modes(to_compare(i,1)).swarm.mode_location,nmmso_state.active_modes(to_compare(i,2)).swarm.mode_location)) < nmmso_state.tol_val
            to_merge = [to_merge; i]; % alas can't preallocate, as don't know the size
        else % otherwise merge if mid region is fitter
            % evaluate exact mid point between modes, and add to mode 2
            % history
            
            mid_loc = 0.5*(nmmso_state.active_modes(to_compare(i,1)).swarm.mode_location-nmmso_state.active_modes(to_compare(i,2)).swarm.mode_location)+nmmso_state.active_modes(to_compare(i,2)).swarm.mode_location;
            
            % little sanity check
            if (sum(mid_loc < mn)> 0 || sum(mid_loc > mx)>0)
                error('mid point out of range!');
            end
            
            nmmso_state.active_modes(to_compare(i,2)).swarm.new_location = mid_loc;
            [nmmso_state,mode_shift,y] = evaluate_mid(nmmso_state,to_compare(i,2),problem_func,problem_function_params);
            if mode_shift==1 % better than mode 2 current mode, so merge
                nmmso_state.M_loc(to_compare(i,2),:)= nmmso_state.active_modes(to_compare(i,2)).swarm.mode_location;
                nmmso_state.V_loc(to_compare(i,2))= nmmso_state.active_modes(to_compare(i,2)).swarm.mode_value;
                to_merge = [to_merge; i];
                nmmso_state.active_modes_changed(to_compare(i,2))=1; % track that the mode value has improved
            elseif (nmmso_state.active_modes(to_compare(i,1)).swarm.mode_value < y) % better than mode 1 current mode, so merge
                to_merge = [to_merge; i];
            end
            number_of_mid_evals = number_of_mid_evals+1;
        end
    end
    % merge those marked pairs, and flag the lower one for deletion
    delete_index= zeros(size(to_merge));
    for i=1:length(to_merge)
        if (to_compare(to_merge(i),2)==to_compare(to_merge(i),1))
            error('indices sould not be equal');
        end
        % if peak of mode 1 is higher than mode 2, then replace
        if (nmmso_state.active_modes(to_compare(to_merge(i),1)).swarm.mode_value > nmmso_state.active_modes(to_compare(to_merge(i),2)).swarm.mode_value)
            delete_index(i) = to_compare(to_merge(i),2);
            nmmso_state.active_modes(to_compare(to_merge(i),1)).swarm = merge_swarms_together(nmmso_state.active_modes(to_compare(to_merge(i),1)).swarm,nmmso_state.active_modes(to_compare(to_merge(i),2)).swarm);
            nmmso_state.active_modes_changed(to_compare(i,1))=1; % track that the mode value has merged and should be compared again
        else
            delete_index(i) = to_compare(to_merge(i),1);
            nmmso_state.active_modes(to_compare(to_merge(i),2)).swarm = merge_swarms_together(nmmso_state.active_modes(to_compare(to_merge(i),2)).swarm,nmmso_state.active_modes(to_compare(to_merge(i),1)).swarm);
            nmmso_state.active_modes_changed(to_compare(i,2))=1; % track that the mode value has merged and should be compared again
        end
    end
    
    % remove one of the merged pair
    prev_merge=-1;
    delete_index=sort(delete_index);
    for i=length(delete_index):-1:1
        if (delete_index(i)~= prev_merge) % if not duplicated
            prev_merge=delete_index(i);
            nmmso_state.active_modes(delete_index(i))=[];
            nmmso_state.M_loc(delete_index(i),:)=[];
            nmmso_state.V_loc(delete_index(i))=[];
            nmmso_state.converged_modes(delete_index(i))=[];
            nmmso_state.active_modes_changed(delete_index(i))=[];
        end
    end
end
if length(nmmso_state.active_modes)==1
    nmmso_state.active_modes(1).swarm.dist = min(mx-mn); % only one mode, so choose arbitary dist for it (smallest design dimension)
end

%--------------------------------------------------------------------------
function [nmmso_state,mode_shift,y] = evaluate(nmmso_state,chg,problem_func,problem_function_params)


y = feval(problem_func, nmmso_state.active_modes(chg).swarm.new_location,problem_function_params);
mode_shift=0;
if (y > nmmso_state.active_modes(chg).swarm.mode_value)
    nmmso_state.active_modes(chg).swarm.mode_location = nmmso_state.active_modes(chg).swarm.new_location;
    nmmso_state.active_modes(chg).swarm.mode_value = y;
    mode_shift = 1;
end

nmmso_state.active_modes(chg).swarm.history_locations(nmmso_state.active_modes(chg).swarm.shifted_loc,:) =  nmmso_state.active_modes(chg).swarm.new_location;
nmmso_state.active_modes(chg).swarm.history_values(nmmso_state.active_modes(chg).swarm.shifted_loc) = y;
% if better than personal best for swarm member - then replace
if (y > nmmso_state.active_modes(chg).swarm.pbest_values(nmmso_state.active_modes(chg).swarm.shifted_loc))
    nmmso_state.active_modes(chg).swarm.pbest_values(nmmso_state.active_modes(chg).swarm.shifted_loc) = y;
    nmmso_state.active_modes(chg).swarm.pbest_locations(nmmso_state.active_modes(chg).swarm.shifted_loc,:) = nmmso_state.active_modes(chg).swarm.new_location;
end


nmmso_state.X(nmmso_state.index,:) = nmmso_state.active_modes(chg).swarm.new_location;
nmmso_state.Y(nmmso_state.index) = y;
nmmso_state.index = nmmso_state.index+1;

%--------------------------------------------------------------------------
function [nmmso_state,mode_shift,y] = evaluate_mid(nmmso_state,chg,problem_func,test_function_params)

% new_location is the only solution thus far in mode, so by definition is
% also the mode estimate, and the only history thus far

y = feval(problem_func, nmmso_state.active_modes(chg).swarm.new_location,test_function_params);
mode_shift=0;

if (y > nmmso_state.active_modes(chg).swarm.mode_value)
    nmmso_state.active_modes(chg).swarm.mode_location = nmmso_state.active_modes(chg).swarm.new_location;
    nmmso_state.active_modes(chg).swarm.mode_value = y;
    mode_shift = 1;
end

nmmso_state.X(nmmso_state.index,:) = nmmso_state.active_modes(chg).swarm.new_location;
nmmso_state.Y(nmmso_state.index) = y;
nmmso_state.index = nmmso_state.index+1;


%--------------------------------------------------------------------------
function swarm1 = merge_swarms_together(swarm1,swarm2)

% merges swarm1 contents into swarm2, keeping the best elements of both

n1 = swarm1.number_of_particles;
n2 = swarm2.number_of_particles;
max_size = size(swarm1.history_locations,1);

if (n1+n2<=max_size)
    swarm1.number_of_particles = n1+n2;
    % simplest situation, where the combined active members of both
    % populations are below the total size they can grow to
    swarm1.history_locations(n1+1:n1+n2,:) = swarm2.history_locations(1:n2,:); % current locations of swarm
    swarm1.history_values(n1+1:n1+n2) = swarm2.history_values(1:n2); % current values of swarm
    
    swarm1.pbest_locations(n1+1:n1+n2,:) = swarm2.pbest_locations(1:n2,:); % current best locations of swarm
    swarm1.pbest_values(n1+1:n1+n2) = swarm2.pbest_values(1:n2); % current best locations of swarm
    
    swarm1.velocities(n1+1:n1+n2,:) = swarm2.velocities(1:n2,:); % current velocities of swarm
else
    % select best out of combined population, based on current location
    % (rather than pbest)
    
    swarm1.number_of_particles = max_size;
    temp_h_loc = [swarm1.history_locations(1:n1,:); swarm2.history_locations(1:n2,:)];
    temp_h_v = [swarm1.history_values(1:n1); swarm2.history_values(1:n2)];
    
    temp_p_loc = [swarm1.pbest_locations(1:n1,:); swarm2.pbest_locations(1:n2,:)];
    temp_p_v = [swarm1.pbest_values(1:n1); swarm2.pbest_values(1:n2)];
    temp_vel = [swarm1.velocities(1:n1,:); swarm2.velocities(1:n2,:)];
    
    [~,I] = sort(temp_h_v,'descend');
    swarm1.history_locations = temp_h_loc(I(1:max_size),:);
    swarm1.history_values = temp_h_v(I(1:max_size),:);
    swarm1.pbest_locations = temp_p_loc(I(1:max_size),:);
    swarm1.pbest_values = temp_p_v(I(1:max_size),:);
    swarm1.velocities = temp_vel(I(1:max_size),:);
end

%--------------------------------------------------------------------------
function [nmmso_state, cs] = increment_swarm(nmmso_state,chg,mn,mx,swarm_size)

cs =0;
new_location = mn-1;

d = nmmso_state.active_modes(chg).swarm.dist;

shifted =0;
omega=0.1;
reject=0;
r = randperm(swarm_size); % select a particle at random to move

while (sum(new_location < mn)> 0 || sum(new_location > mx)>0)
    % if swarm not yet at capacity, simply add a new particle
    if nmmso_state.active_modes(chg).swarm.number_of_particles < swarm_size
        new_location = nmmso_state.active_modes(chg).swarm.mode_location + uniform_sphere_points(1,length(new_location))*(d/2);
    else % otherwise move an existing particle
        shifted =1;
        nmmso_state.active_modes(chg).swarm.shifted_loc = r(1);
        temp_velocity = omega*nmmso_state.active_modes(chg).swarm.velocities(nmmso_state.active_modes(chg).swarm.shifted_loc,:) ...
            + 2.0 * rand(size(new_location)).*(nmmso_state.active_modes(chg).swarm.mode_location - nmmso_state.active_modes(chg).swarm.history_locations(nmmso_state.active_modes(chg).swarm.shifted_loc,:)) ...
            + 2.0 * rand(size(new_location)).*(nmmso_state.active_modes(chg).swarm.pbest_locations(nmmso_state.active_modes(chg).swarm.shifted_loc,:) - nmmso_state.active_modes(chg).swarm.history_locations(nmmso_state.active_modes(chg).swarm.shifted_loc,:));
        if (reject>20)
            % if we keep rejecting, then put at extreme any violating
            % design parameters
            I_max= find(((nmmso_state.active_modes(chg).swarm.history_locations(nmmso_state.active_modes(chg).swarm.shifted_loc,:) + temp_velocity)>mx)==1);
            I_min= find(((nmmso_state.active_modes(chg).swarm.history_locations(nmmso_state.active_modes(chg).swarm.shifted_loc,:) + temp_velocity)<mn)==1);
            if isempty(I_max)==0
                temp_velocity(I_max)= rand(1,length(I_max)).*(mx(I_max)-nmmso_state.active_modes(chg).swarm.history_locations(nmmso_state.active_modes(chg).swarm.shifted_loc,I_max));
            end
            if isempty(I_min)==0
                temp_velocity(I_min)= rand(1,length(I_min)).*((nmmso_state.active_modes(chg).swarm.history_locations(nmmso_state.active_modes(chg).swarm.shifted_loc,I_min)-mn(I_min))*-1);
            end
        end
        new_location = nmmso_state.active_modes(chg).swarm.history_locations(nmmso_state.active_modes(chg).swarm.shifted_loc,:) + temp_velocity;
        reject=reject+1;
    end
end
reject =0;
if (shifted ==1) % if moved, update velocity with that used
    nmmso_state.active_modes(chg).swarm.velocities(nmmso_state.active_modes(chg).swarm.shifted_loc,:) = temp_velocity;
else % otherwise initialise velocity in sphere based on distance from gbest to next closest mode
    nmmso_state.active_modes(chg).swarm.number_of_particles = nmmso_state.active_modes(chg).swarm.number_of_particles+1;
    nmmso_state.active_modes(chg).swarm.shifted_loc = nmmso_state.active_modes(chg).swarm.number_of_particles;
    temp_vel = mn-1;
    while (sum(temp_vel < mn)> 0 || sum(temp_vel > mx)>0)
        temp_vel = uniform_sphere_points(1,length(new_location))*(d/2);
        reject = reject+1;
        if reject>20 % resolve if keep rejecting
            temp_vel = rand(size(new_location)).*(mx-mx) + mn;
        end
    end
    nmmso_state.active_modes(chg).swarm.velocities(nmmso_state.active_modes(chg).swarm.shifted_loc,:) = temp_vel;
    
end
nmmso_state.active_modes(chg).swarm.new_location = new_location;

%--------------------------------------------------------------------------
function [nmmso_state,number_of_new_locations] = evaluate_new_locations(nmmso_state,problem_func,problem_function_params,I)


nmmso_state.active_modes_changed = zeros(length(nmmso_state.active_modes),1); % at this point should be unflagged
for i=1:length(I)
    [nmmso_state,mode_shift] = evaluate(nmmso_state,I(i),problem_func,problem_function_params);
    if mode_shift==1
        nmmso_state.active_modes_changed(I(i)) = 1;
        nmmso_state.M_loc(I(i),:) = nmmso_state.active_modes(I(i)).swarm.new_location;
        nmmso_state.V_loc(I(i)) = nmmso_state.active_modes(I(i)).swarm.mode_value;
        nmmso_state.active_modes(I(i)).swarm.less_fit_move=0;
    end
end
number_of_new_locations = length(I);


%--------------------------------------------------------------------------
function [nmmso_state,number_of_new_modes] = evolve(nmmso_state,problem_func,mn,mx,problem_function_params,max_evol,swarm_size)

n = length(nmmso_state.active_modes);

if n>max_evol
    if rand()<0.5
        [~, I] = sort(nmmso_state.V_loc,'descend');
    else
        I=1:n;
    end
    I = I(1:max_evol);
    n = max_evol;
else
    I = 1:n;
end

II = randperm(n);
% uniform crossover of two mode elements, either fittest two, or random two
R = UNI(nmmso_state.active_modes(I(II(1))).swarm.mode_location,nmmso_state.active_modes(I(II(2))).swarm.mode_location);


nmmso_state.M_loc = [nmmso_state.M_loc; R];

swarm.new_location = R;
[swarm,nmmso_state] = evaluate_first(swarm,problem_func,problem_function_params,nmmso_state,swarm_size,mn,mx);
nmmso_state.V_loc = [nmmso_state.V_loc; swarm.mode_value];
nmmso_state.active_modes(end+1).swarm = swarm;

% mark these as new
nmmso_state.active_modes_changed = [nmmso_state.active_modes_changed; 1];
nmmso_state.converged_modes = [nmmso_state.converged_modes; 0];
number_of_new_modes = 1;


%--------------------------------------------------------------------------
function [nmmso_state,number_of_new_samples] = hive(nmmso_state,problem_func,mn,mx,problem_function_params,max_evol,swarm_size)

number_of_new_samples=0;

LL = length(nmmso_state.active_modes);
fit_I = randperm(LL);

limit = min(max_evol,LL);

I2 = fit_I(1:limit);
CI = zeros(length(I2));
% first identify those swarms who are at capacity, and therefore may be
% considered for splitting off a member
for i=1:length(I2);
    if nmmso_state.active_modes(i).swarm.number_of_particles >= swarm_size
        CI(i)=1;
    end
end
CI = find(CI==1);
% only check on full swarms
if isempty(CI)==0
    % select swarm at random
    r = randperm(length(CI));
    r = CI(r(1));
    % select an active swarm member at random
    k = randperm(nmmso_state.active_modes(r).swarm.number_of_particles);
    k = k(1);
    R = nmmso_state.active_modes(r).swarm.history_locations(k,:);
    R_v = nmmso_state.active_modes(r).swarm.history_values(k,:);
    
    % only look at splitting off member who is greater than tol_value
    % distance away -- otherwise will be merged right in again at the
    % next iteration
    if sqrt(dist2(R,nmmso_state.active_modes(r).swarm.mode_location))>nmmso_state.tol_val 
        
        mid_loc = 0.5*(nmmso_state.active_modes(r).swarm.mode_location-R)+R;
        
        swarm.new_location = mid_loc;
        [swarm,nmmso_state] = evaluate_first(swarm,problem_func,problem_function_params,nmmso_state,swarm_size,mn,mx);
        mid_loc_val = swarm.mode_value;
        % if valley between, then hive off the old swarm member to create new swarm
        if swarm.mode_value < R_v
            reject=0;
            % allocate new swarm
            swarm.mode_location = R; %gbest location
            swarm.mode_value = R_v; %gbest value
            
            swarm.history_locations(1,:) = R;
            swarm.history_values(1) = R_v;
            
            swarm.pbest_locations(1,:) = R;
            swarm.pbest_values(1) = R_v;
            
            nmmso_state.M_loc = [nmmso_state.M_loc; R];
            nmmso_state.V_loc = [nmmso_state.V_loc; R_v];
            
            nmmso_state.active_modes(end+1).swarm = swarm;
            
            nmmso_state.active_modes_changed = [nmmso_state.active_modes_changed; 1];
            nmmso_state.converged_modes = [nmmso_state.converged_modes; 0];
            
            % remove from existing swarm and replace with mid eval
            d = sqrt(dist2(nmmso_state.active_modes(r).swarm.mode_location,R));
            
            nmmso_state.active_modes(r).swarm.history_locations(k,:) = mid_loc;
            nmmso_state.active_modes(r).swarm.history_values(k,:) = mid_loc_val;
            
            nmmso_state.active_modes(r).swarm.pbest_locations(k,:) = mid_loc;
            nmmso_state.active_modes(r).swarm.pbest_values(k,:) = mid_loc_val;
            
            temp_vel = mn-1;
            while (sum(temp_vel < mn)> 0 || sum(temp_vel > mx)>0)
                temp_vel = uniform_sphere_points(1,length(R))*(d/2);
                reject = reject+1;
                if reject>20 % resolve repeated rejection
                    temp_vel = rand(size(R)).*(mx-mn) + mn;
                end
            end
            nmmso_state.active_modes(r).swarm.velocities(k,:) = temp_vel;
        else
            if swarm.mode_value > nmmso_state.active_modes(r).swarm.mode_value
                % discovered better than original, so replace mode accordingly
                nmmso_state.active_modes(r).swarm.mode_value = swarm.mode_value;
                nmmso_state.active_modes(r).swarm.mode_location = swarm.mode_location;
            end
        end
        
        number_of_new_samples = number_of_new_samples+1;
    end
    
end

%--------------------------------------------------------------------------
function [nmmso_state,number_rand_modes] = random_new(nmmso_state,problem_func,mn,mx,problem_function_params,swarm_size)

number_rand_modes = 1;
x = rand(size(mx)).*(mx-mn)+mn;

nmmso_state.active_modes_changed = [nmmso_state.active_modes_changed; ones(number_rand_modes,1)];
nmmso_state.converged_modes = [nmmso_state.converged_modes; zeros(number_rand_modes,1)];
swarm.new_location = x(1,:);

[swarm,nmmso_state] = evaluate_first(swarm,problem_func,problem_function_params,nmmso_state,swarm_size,mn,mx);
nmmso_state.active_modes(end+1).swarm = swarm;
nmmso_state.M_loc = [nmmso_state.M_loc; x];
nmmso_state.V_loc = [nmmso_state.V_loc; nmmso_state.active_modes(end).swarm.mode_value];

%--------------------------------------------------------------------------

function [x_c, x_d] = UNI(x1,x2)
% simulated binary crossover
l = length(x1);
x_c =x1;
x_d = x2;
r = find(rand(l,1)>0.5);
if isempty(r)==1 % ensure at least one swapped
    r = randperm(l);
    r=r(1);
end
x_c(r) = x2(r);
x_d(r) = x1(r);

%--------------------------------------------------------------------------

function X = uniform_sphere_points(n,d)

% X = uniform_sphere_points(n,d)
%
%function generates n points unformly within the unit sphere in d dimensions


z= randn(n,d);

r1 = sqrt(sum(z.^2,2));

X=z./repmat(r1,1,d);
r=rand(n,1).^(1/d);

X = X.*repmat(r,1,d);

