function [mode_loc_before,mode_y_before,evaluations_before,nmmso_state, ...
    mode_loc_after,mode_y_after,evaluations_after] =  NMMSO( ...
    swarm_size, problem_func,problem_function_params, max_evaluations, ...
    mn,mx,max_evol,tol_val)



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
%
% OPTIONAL ARGUMENTS
%
% max_evols = maximum number of swarms to update in a generation. If not
%   provided this is set at 100
% tol_val = tolerance value for merging automatically (default 10^-6)
%
% OUTPUTS
%
% Due to the algorithm design (dynamic populations) the final generation
% may exceed the alloted maximum number of evaluations. As such the final
% peak estimate state and the penultimate peak estimate state are returned 
% (with the corresponding evaluation number tracked)
%
% mode_loc_before = design space location of penultimate mode estimates (swarm
%   gbests), note that at least one is likely to be very poor due to the
%   new swarm spawning at the end of each generation, and that these will
%   be a combination of both global and local mode estimate
% mode_y_before = function evalutions corresponding to the mode estimates
% evaluations_before = number of problem function evaluations at end of
%   penultimate generation
% mode_loc_after = design space location of mode estimates at end 
% mode_y_after = function evalutions corresponding to the mode estimates
% evaluations_after = number of problem function evaluationsat end
%
% nmmso_state = structure holding the state of the swarms. Unless you want
%   to pick apart the details of how the algorithm searchs the space, then 
%   the only two elements you will probably be interested in are X and Y
%   which are preallocated matrices to hold all locations visited 
%   (therefore nmmso_state.X(1:evaluations,:) will hold all the design
%   space locations visited by the optimiser thus far. The final version is
%   returned (can be quite large)


if exist('max_evol','var')==0
    display('default max_eval used, set at 100');
    max_evol=100;
end

if max_evol<=0
    display('Max_eval cannot be negative or zero, default max_eval used, set at 100');
    max_evol=100;
end

if exist('tol_value','var') ==0
    tol_val = 10^-6;
end


% at start no evaluations used, and NMMSO state is empty
mode_loc_after=[];
mode_y_after=[];
evaluations_after=[];
nmmso_state=[];
evaluations_after=0;

while (evaluations_after < max_evaluations)
    mode_loc_before=mode_loc_after;
    mode_y_before=mode_y_after;
    evaluations_before=evaluations_after;
    [mode_loc_after,mode_y_after,evaluations_after,nmmso_state] =  NMMSO_iterative( ...
        swarm_size, problem_func,problem_function_params, max_evaluations, ...
        mn,mx,evaluations_after,nmmso_state,max_evol,tol_val);
end


