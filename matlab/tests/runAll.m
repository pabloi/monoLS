function runAll()
%Run every MATLAB/Octave test (each test is a function that errors on failure).
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..'));
files = dir(fullfile(here, 'test*.m'));
failed = {};
ran = 0;
for i = 1:numel(files)
    name = files(i).name(1:end-2);
    ran = ran + 1;
    try
        feval(name);
    catch err
        failed{end+1} = name; %#ok<AGROW>
        fprintf(2, '%s FAILED: %s\n', name, err.message);
    end
end
if ~isempty(failed)
    error('runAll:failed', '%d test file(s) failed', numel(failed));
end
fprintf('runAll: %d test files passed\n', ran);
end
