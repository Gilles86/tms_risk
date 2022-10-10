
function physio = prepare_retroicor(subject, session)
    subject
    session
   
%     %% Create default parameter structure with all fields
     physio = tapas_physio_new();
% 
%     %% Individual Parameter settings. Modify to your need and remove default settings

    if strcmp(subject, '33') & strcmp(session, '1')
        runs = [1 3 4 5];
    else
        runs = 1:6;
    end
   
    task = 'task'; 

    disp(runs)
    
    for run = runs
        close all;
        retroicor(subject, session, task, run)
    end
end