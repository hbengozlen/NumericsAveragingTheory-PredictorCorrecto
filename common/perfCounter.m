function out = perfCounter(name, action, field, value)
%PERFCOUNTER Generic named performance-counter accumulator, without a
%   global variable. State lives in this function's own persistent
%   registry, keyed by NAME, so unrelated call sites (solveVelocityFFT's
%   setup/Av/Pinv, solveVelocity's build/solve, any future solver's own
%   counters) can't accidentally read or clobber each other's state the
%   way a bare global variable could.
%
%   perfCounter(name, 'reset')                  -- zero out this name's counters
%   perfCounter(name, 'add', field, value)      -- counters.(field) += value
%                                                   (field auto-initializes
%                                                   to 0 on first use)
%   s = perfCounter(name, 'get')                -- current struct for this name
%   s = perfCounter(name, 'get', fieldNames)    -- same, but guarantees every
%                                                   name in the fieldNames
%                                                   cell array exists in s
%                                                   (defaulting to 0) --
%                                                   for callers that print a
%                                                   fixed set of fields even
%                                                   when this name was never
%                                                   'add'-ed to this run
%                                                   (e.g. a solver that
%                                                   wasn't called this time)
%
%   Example:
%     perfCounter('solveVelocity', 'reset');
%     tBuild = tic;
%     ...
%     perfCounter('solveVelocity', 'add', 'build', toc(tBuild));
%     s = perfCounter('solveVelocity', 'get', {'build','solve','calls'});
%     fprintf('build: %.3f s\n', s.build);

persistent registry
if isempty(registry)
    registry = containers.Map();
end
if ~isKey(registry, name) || strcmp(action, 'reset')
    registry(name) = struct();
end
counters = registry(name);

switch action
    case 'reset'
        % already reset above
    case 'add'
        if ~isfield(counters, field)
            counters.(field) = 0;
        end
        counters.(field) = counters.(field) + value;
        registry(name) = counters;
    case 'get'
        if nargin >= 3 && ~isempty(field)
            for i = 1:numel(field)
                if ~isfield(counters, field{i})
                    counters.(field{i}) = 0;
                end
            end
        end
    otherwise
        error('perfCounter: unknown action ''%s''', action);
end

out = counters;
end
