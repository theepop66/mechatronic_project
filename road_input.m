%========================================================================================================
% Author: Carl Larsson
% Description: simulates the road input
%========================================================================================================
% Copyright (c) 2023 Carl Larsson
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.
%========================================================================================================
%% Road input
%------------------------------------------------------------------------------------------------
function F_r = road_input(t)
    % Speed bumps
    if t == 10 || t == 40 || t == 80 || t == 90
        F_r = abs(0*sin(0*t + 0) + randi([200000 400000],1,1)*cos(0*t + 0));
    % 0-10km/h
    elseif t<10
        F_r = abs(randi([0 1000],1,1)*sin((0+(2-0)*rand())*t + rand(1)) + randi([0 1000],1,1)*cos((0+(2-0)*rand())*t + rand(1)));
    % 10-20km/h    
    elseif t>10 && t<20
        F_r = abs(randi([0 4000],1,1)*sin((1+(3-1)*rand())*t + rand(1)) + randi([0 4000],1,1)*cos((1+(3-1)*rand())*t + rand(1)));
    % 20-30km/h
    elseif t>20 && t<30
        F_r = abs(randi([0 6000],1,1)*sin((3+(5-3)*rand())*t + rand(1)) + randi([0 6000],1,1)*cos((3+(5-3)*rand())*t + rand(1)));
    % 30-40km/h
    elseif t>30 && t<40
        F_r = abs(randi([0 9000],1,1)*sin((4+(6-4)*rand())*t + rand(1)) + randi([0 9000],1,1)*cos((4+(6-4)*rand())*t + rand(1)));
    % 40-50km/h
    elseif t>40 && t<50
        F_r = abs(randi([0 11000],1,1)*sin((6+(8-6)*rand())*t + rand(1)) + randi([0 11000],1,1)*cos((6+(8-6)*rand())*t + rand(1)));
    % 50-60km/h
    elseif t>50 && t<60
        F_r = abs(randi([0 13000],1,1)*sin((7+(9-7)*rand())*t + rand(1)) + randi([0 13000],1,1)*cos((7+(9-7)*rand())*t + rand(1)));
    % 60-70km/h
    elseif t>60 && t<70
        F_r = abs(randi([0 16000],1,1)*sin((9+(11-9)*rand())*t + rand(1)) + randi([0 16000],1,1)*cos((9+(11-9)*rand())*t + rand(1)));
    % 70-80km/h
    elseif t>70 && t<80
        F_r = abs(randi([0 18000],1,1)*sin((10+(12-10)*rand())*t + rand(1)) + randi([0 18000],1,1)*cos((10+(12-10)*rand())*t + rand(1)));
    % 80-90km/h
    elseif t>80 && t<90
        F_r = abs(randi([0 21000],1,1)*sin((12+(14-12)*rand())*t + rand(1)) + randi([0 21000],1,1)*cos((12+(14-12)*rand())*t + rand(1)));
    % 90-100km/h
    elseif t>90 && t<100
        F_r = abs(randi([0 23000],1,1)*sin((14+(16-14)*rand())*t + rand(1)) + randi([0 23000],1,1)*cos((14+(16-14)*rand())*t + rand(1)));
    % 100-110km/h
    elseif t>100 && t<150
        F_r = abs(randi([0 26000],1,1)*sin((15+(17-15)*rand())*t + rand(1)) + randi([0 26000],1,1)*cos((15+(17-15)*rand())*t + rand(1)));
    % Stop
    else
        F_r = abs(0*sin(0*t + 0) + 0*cos(0*t + 0));
    end
end
%------------------------------------------------------------------------------------------------