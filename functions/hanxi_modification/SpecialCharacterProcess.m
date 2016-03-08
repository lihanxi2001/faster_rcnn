% File Type:     Matlab
% Author:        Hanxi Li {lihanxi2001@gmail.com}
% Creation:      Monday 01/02/2016 10:32.
% Last Revision: Monday 01/02/2016 10:32.

function str_out = SpecialCharacterProcess(str_in)

special_characters = {'+', '?', '.', '*', '(', ')'};
for i = 1 : length(special_characters)
    str_in = regexprep(str_in, ['\', special_characters{i}], ['\\', special_characters{i}]);
end
str_out = str_in;

end


