%% Initialization
%  Initialize the world, Q-table, and hyperparameters
world = 4;
gwinit(world);

eps = 0.5; % Greedy epsilon
eps_decay = 0.001;
min_eps = 0.01;
gamma = 0.95; % Diminishing reward factor
learning_rate = 0.3;
%goal_reward = 1;

num_episodes = 10000;
max_actions = 100;

q_table = zeros(gwstate().xsize, gwstate().ysize,4);
q_table(:,1,2) = -Inf;
q_table(:,gwstate().ysize,1) = -Inf;
q_table(1,:,4) = -Inf;
q_table(gwstate().xsize,:,3) = -Inf;


%% Training loop
%  Train the agent using the Q-learning algorithm.

for episode = 1:num_episodes
    %reset
    %pos = get_pos_from_gwinit
    
    if mod(episode,100) == 0
        disp(episode)
    end
    
    for i = 1:max_actions
        while 1     
            old_x = gwstate().pos(2);
            old_y = gwstate().pos(1);

            if rand(1) < eps % Take random action 

                action = randperm(4,1);  

            else % Take opt action

                [~,action] = max(q_table(old_x,old_y,:),[],3); 

            end

            gwaction(action);
            if gwstate().feedback > 0 
                disp("pos feedback")
                gwstate().feedback
            end
            
            if gwstate().isvalid
                %pause(0.1)
                %gwdraw();
                break
            end
            
        end
       
        if gwstate().isterminal % Finding the target
           
           %reward = goal_reward;
           reward = gwstate().feedback;
           if reward >= 0 
               disp("pos reward")
           end
           value = max(q_table(gwstate().pos(2),gwstate().pos(1),:),[],3); 
        
           q_table(old_x, old_y, action) = q_table(old_x, old_y, action)*(1-learning_rate) + learning_rate*(reward + gamma*value);
            
           %disp("God yub")
           %q_table
           %pause(10)
           gwinit(world);
           break
        else
        
        reward = gwstate().feedback;
        value = max(q_table(gwstate().pos(2),gwstate().pos(1),:),[],3); 
        
        q_table(old_x, old_y, action) = q_table(old_x, old_y, action)*(1-learning_rate) + learning_rate*(reward + gamma*value);
        %disp(reward);
        
        end
        
        if i >= max_actions
           %disp("MAX ACTIONS")
           gwinit(world);
        end
    end 
    
    eps = eps - eps_decay; %Move?
    if eps < min_eps 
        eps = min_eps; 
    end
       
end

figure(1)
imagesc(getvalue(q_table))
%{
figure(2)
imagesc(q_table(:,:,2))
figure(3)
imagesc(q_table(:,:,3))
figure(4)
imagesc(q_table(:,:,4))
figure(5)
%}
figure(2)
P = getpolicy(q_table);
e = num_episodes;
gwdraw(e, P)


%% Test loop
%  Test the agent (subjectively) by letting it use the optimal policy
%  to traverse the gridworld. Do not update the Q-table when testing.
%  Also, you should not explore when testing, i.e. epsilon=0; always pick
%  the optimal action.
%{
good = 0;
for episode = 1:1000
    %reset
    %pos = get_pos_from_gwinit
    
    for i = 1:max_actions
        while 1     
            old_x = gwstate().pos(2);
            old_y = gwstate().pos(1);

            
            [~,action] = max(q_table(old_x,old_y,:),[],3); 

          

            gwaction(action);
           
            
            if gwstate().isvalid
                pause(0.1)
                gwdraw();
                break
            end
            
        end
        
        if gwstate().isterminal % Finding the target
           good = good +1;
           %disp("God yub")
           gwinit(world);
           break
        end
        
        if i >= max_actions
           %disp("MAX ACTIONS")
           gwinit(world);
        
        end
    end
end
disp("acc: ")
disp(good/1000)
%}

