function [evals, swarm, X, Y] = ...
    MSSPSO_iterative(swarm_size, swarm_num, max_evaluations, problem_func,...
    problem_func_params, param_num, mn, mx, evals, swarm)


% Implementation of the algorithm described in:
% "Multi-Sub-Swarm Particle Swarm Optimization Algorithm for Multimodal 
% Function Optimization" 
% by Jun Zhang et al.
% published in Proceedings of the IEEE Congress on Evolutionary Computation, 
% pages 3215-3220, 2007
%
% Implementation provided corresponds to that used in:
%"Running Up Those Hills: Multi-Modal Search with the Niching Migratory 
% Multi-Swarm Optimiser"
% by Jonathan E. Fieldsend
% published in Proceedings of the IEEE Congress on Evolutionary Computation, 
% pages 2593-2600, 2014
%
% Please reference both papers if you undertake work utilising this code.
%
% Implementation (c) by Jonathan Fieldsend, University of Exeter, 2014
%
% Assumes function maximisation
%
% REQUIRED ARGUMENTS
%
% pop_size = population size
% max_evaluations = maximum number of evaluations to be taken through the
%   problem function
% problem_func = string containing function to be optimised
% problem_func_params = meta-parameters needed by test function (distinct
%   from optimisation (design) parameters
% param_num = number of design parameters
% mn = minimum design parameter values (a vector with param_num elements)
% mx = maximum design parameter values (a vector with param_num elements)
%
% OPTIONAL ARGUMENTS
%
% evals = If evals argument is zero, or absent, then the optimiser starts from a 
% random population, otherwise will increment from the population specified 
% in the following optional arguments
% swarm = structure containing swarms (output by the algorithm and can be 
% fed back in when used iteratively)
%
% OUTPUTS
%
% evals = number of function evaluations used by function return
% swarm = structure containing state of swarm(s) at 'evals' function
% evaluations
% X = matrix containing location of all the swarms pbests (first half) 
% and current particle locations (second half)
% Y = function response at X locations 

if exist('evals','var')==0
   evals = 0;
   fprintf('evals set at zero');
end

if evals<0
   error('number of completed evalutions cannot be negative');
end

if swarm_size<1
   error('swarm size must be at least 1 member');
end

if swarm_num<1
   error('swarm number must at least be 1');
end


if max_evaluations<1
   error('max_evaluations must at least be 1');
end

if exist('swarm','var')==0
    if evals>0
       error(strcat('swarm structure areument not passed in, however number of', ...
       'previous evaluations is claimed to be positive')); 
    end
end


% set up algorithm as in paper, paper states sample array is [0.01,0.09],
% however as this is not symetric it does not make sense (without reference
% to which niches are at either end), so it is assumed that is is a typo,
% and [0.01, 0.99] is meant (i.e, 1% away from each peak). This indeed
% makes the algorithm behave much better.

sample_array = [0.01, 0.99];
inertia = 0.729;
C1 = 1.49445;
C2 = 1.49445;
range = mx-mn;
Vmax = range;
% pre-calculate scaling factors for valid ranges

range_mat = repmat(range,swarm_size,1);
mn_mat = repmat(mn,swarm_size,1);


% start from scratch, or use optional arguments
if evals==0
    % create and initialise the sub-swarms
    
    % initialise swarms
    for i=1:swarm_num
        [swarm] = initialise_sub_swarm(swarm, i, mn_mat, range_mat, ...
            swarm_size, problem_func, problem_func_params, param_num);
    end
    evals = swarm_num * swarm_size;
end

% until we hit the maximium number of evaluations
if (evals < max_evaluations)
     
    %print_best(swarm,'start');
    % determine which is the _actual_ best in each swarm, otherwise you can
    % lose the best solutions
    for i=1:swarm_num
        swarm = determine_gbest(swarm,i);
    end
    % for each sub-swarm, compare to all others and mark the losers
    for i=1:swarm_num
        % if the best particle of the swarm is located in the same niche as
        % a different swarm, mark the loser and winner
        loser = -1;
        for j=1:swarm_num
            if (i~=j)
                [valley, calls] = hill_valley(swarm(i).pbest(swarm(i).gbest_index,:), ...
                    swarm(j).pbest(swarm(j).gbest_index,:), swarm(i).pbest_eval(swarm(i).gbest_index), ...
                    swarm(j).pbest_eval(swarm(j).gbest_index), sample_array, ...
                    problem_func, problem_func_params);
                evals = evals + calls;
                if (valley==0)
                    % on same hill, so compete, based on actual fitness
                    if swarm(i).pbest_eval(swarm(i).gbest_index) > swarm(j).pbest_eval(swarm(j).gbest_index)
                        loser = j;
                    else
                        loser = i;
                    end
                    break;
                end
            end
        end
        if (loser~=-1)
            % there is a loser, so reinitialise it
            [swarm] = initialise_sub_swarm(swarm, loser, mn_mat, range_mat, ...
                swarm_size, problem_func, problem_func_params, param_num);
            evals = evals + swarm_size;
        end
    end
    
    %print_best(swarm,'post-battle');
    % for every particle and remembered particle position of each sub-swarm
    for i=1:swarm_num
        % penalise those which stray onto other peaks, set pbest, determine
        % gbest
        [swarm,evals] = penalise_strays(swarm, i, swarm_size, ...
                    sample_array, problem_func, problem_func_params, evals);
    end
    
    %print_best(swarm,'pbest updated');
    % train each subswarm
    for i=1:swarm_num
       [swarm, evals] = move_and_evaluate_particles(swarm, i, inertia, ...
           C1, C2, mx, mn, Vmax, problem_func, problem_func_params, param_num, ...
           swarm_size, evals); 
    end
    
    %print_best(swarm,'evolved');
    
    X = zeros(swarm_size*swarm_num*2,param_num);
    Y = zeros(swarm_size*swarm_num*2,1);
    index =1;
    for i=1:swarm_num
        X(index:index+swarm_size-1,:) = swarm(i).pbest;
        Y(index:index+swarm_size-1) = swarm(i).pbest_eval;
        index=index+swarm_size;
    end
    
    for i=1:swarm_num
        X(index:index+swarm_size-1,:) = swarm(i).particles;
        Y(index:index+swarm_size-1) = swarm(i).particles_eval;
        index=index+swarm_size;
    end
    fprintf('Evals %d, best solution %f\n',evals, max(Y));
end

%------------
function print_best(swarm,message)

Y = zeros(length(swarm),1);

for i=1:length(swarm)
    Y(i) = swarm(i).pbest_eval(swarm(i).gbest_index);
end

fprintf('Best solution %s %f\n', message, max(Y));

%------------
function swarm = determine_gbest(swarm,i)

[vpart,Ipart] = sort(swarm(i).particles_eval,'descend');
[vbest,Ibest] = sort(swarm(i).pbest_eval,'descend');
if vpart>vbest
    % a new particle location is actually better than the best pbest
    % stored, so update
    swarm(i).pbest_fitness(Ipart(1)) = swarm(i).particles_eval(Ipart(1));
    swarm(i).pbest(Ipart(1),:) = swarm(i).particles(Ipart(1),:);
    swarm(i).gbest_index = Ipart(1);
else
    % a pbest is still best, but ensure tha actual value is stored
    swarm(i).pbest_fitness(Ibest(1)) = swarm(i).pbest_eval(Ibest(1));
    swarm(i).gbest_index = Ibest(1);
end

%------------
function [swarm] = initialise_sub_swarm(swarm, i, mn_mat, range_mat, ...
    swarm_size, problem_func, problem_func_params, param_num)


swarm(i).particles = mn_mat + rand(swarm_size,param_num).*range_mat;
swarm(i).pbest = swarm(i).particles;
swarm(i).particles_eval = zeros(swarm_size,1);
swarm(i).velocities = mn_mat + rand(swarm_size,param_num).*range_mat;
for j=1:swarm_size
    swarm(i).particles_eval(j) = feval(problem_func, swarm(i).particles(j,:), problem_func_params);
end
swarm(i).pbest_eval = swarm(i).particles_eval;
swarm(i).pbest_legal_niche = ones(swarm_size,1);
swarm(i).particles_legal_niche = ones(swarm_size,1); % to begin with all evals are assumed on the same peak
[~,swarm(i).gbest_index] = max(swarm(i).pbest_eval);

%------------
function [swarm,evals] = penalise_strays(swarm, i, swarm_size, ...
                    sample_array, problem_func, problem_func_params, evals)

% reduce the fitness of any particle that has strayed onto another peak.
% 
% Unfortunately the penalty function is mentioned but never defined in the
% 2007 paper this is based on. The authors use the penalty to try and
% prevent the pbest being updated with locations which are not on the same
% niche as the gbest and likewise penalise current pbest if they are not on
% niche as gbest (to prevent replacing). The same effect can be
% accomplished by simply flagging, which is not susceptible to scale
% dependent penalty functions either. This approach is therefore taken here



% if the particle is not on same niche as gbest, then non_tracking(k)>0
for k=1:swarm_size
    [non_tracking,calls] = hill_valley(swarm(i).pbest(swarm(i).gbest_index,:), ...
        swarm(i).particles(k,:), swarm(i).pbest_eval(swarm(i).gbest_index), ...
        swarm(i).particles_eval(k), sample_array, ...
        problem_func, problem_func_params);
    % if on same peak as gbest, then flag as legal to update pbest,
    % otherwise flagged with 0
    swarm(i).particles_legal_niche(k) = (non_tracking==0);
    evals = evals+calls;
end


updated = zeros(swarm_size,1);
% check if pbests need updating -- use the flag as a penalty proxy to
% prevent pbest being replaced with a location 'off-peak' (as a penalty
% would downgrade the particle location fitness to prevent replacement)
for j=1:swarm_size
    if swarm(i).particles_legal_niche(j)
        % if current location is on niche, and better than pbest, update
        % pbest
        if swarm(i).pbest_eval(j) < swarm(i).particles_eval(j)
            swarm(i).pbest(j,:) = swarm(i).particles(j,:);
            swarm(i).pbest_eval(j) = swarm(i).particles_eval(j);
            swarm(i).pbest_legal_niche(j) = swarm(i).particles_legal_niche(j);
            updated(j)=1;
        end
    end
end

[~,swarm(i).gbest_index] = max(swarm(i).pbest_eval);

%------------
function [swarm, evals] = move_and_evaluate_particles(swarm, i, inertia, ...
           C1, C2, mx, mn, Vmax, problem_func, problem_func_params, ...
           param_num, swarm_size, evals)

       
for j=1:swarm_size
   % calculate velocity
   swarm(i).velocities(j,:) = inertia*swarm(i).particles(j,:) + ... %previous velocity
        C1*rand(1,param_num).*(swarm(i).pbest(j,:) - swarm(i).particles(j,:)) + ... %cognitive guide
        C2*rand(1,param_num).*(swarm(i).pbest(swarm(i).gbest_index,:) - swarm(i).particles(j,:)); % social guide
   % limit velocity
   swarm(i).velocities(j,swarm(i).velocities(j,:)>Vmax) = Vmax(1,swarm(i).velocities(j,:)>Vmax);
   swarm(i).velocities(j,swarm(i).velocities(j,:)<-Vmax) = -Vmax(1,swarm(i).velocities(j,:)<-Vmax);
   % move particle
   swarm(i).particles(j,:) = swarm(i).particles(j,:) + swarm(i).velocities(j,:);
   % boundary condition ensurance not mentioned in paper. Hard limiting
   % used
   swarm(i).particles(j,swarm(i).particles(j,:)>mx) = mx(swarm(i).particles(j,:)>mx);
   swarm(i).particles(j,swarm(i).particles(j,:)<mn) = mn(swarm(i).particles(j,:)<mn);
   % now evaluate
   swarm(i).particles_eval(j) = feval(problem_func, swarm(i).particles(j,:), problem_func_params);
end
evals = evals + swarm_size;

%------------
function [v, calls, pts] = hill_valley(i1, i2, fit1, fit2, samples, problem_func, problem_func_params)

% returns 0 if all sampled points on line between i1 and i2 arguments
% have are equal to or higher than the minimum response of i1 and i2


min_fit = min(fit1,fit2);
v = 0;
calls=0; % keep track of the number of problem function evaluations
pts  = zeros(size(samples));
for j = 1: length(samples);
    % generate point on line between points
    interior_j = i1 +(i2-i1)*samples(j);
    pts(j) = feval(problem_func, interior_j, problem_func_params);
    calls = calls + 1;
    if min_fit > pts(j)
        v = min_fit - pts(j);
        return;
    else
        pts(j) = 0;
    end
end

