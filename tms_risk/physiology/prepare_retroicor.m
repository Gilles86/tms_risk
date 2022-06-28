
function physio = prepare_retroicor(subject, session)
    subject
    session
   
%     %% Create default parameter structure with all fields
     physio = tapas_physio_new();
% 
%     %% Individual Parameter settings. Modify to your need and remove default settings
     runs = 1:6;
     task = 'task'; 

    disp(runs)
    
    for run = runs
        close all;
        retroicor(subject, session, task, run)
    end
end