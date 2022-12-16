subjects = [3 4 5 6 7 9 10 11 18 19 21 25 26 29 30 31 34 35 36 37 45 46 47 50 53 56 59 62 63 67 69 72 74]

subjects = [1 2]

subjects = [30 31 34 35 36 37 45 46 47 50 53 56 59 62 63 67 69 72 74]


for sub = subjects

    disp(sub);
    sub_str = sprintf('%02d', sub);
    disp(sub_str);
    
    prepare_retroicor(sub_str, '1');
end

