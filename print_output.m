function print_output(a)
if (size(a)!=[1 7])
    printf('error: incorrect argument size')
    return
end

c = ['_', '|', '_', '|', '|', '_', '|'];
c(find(!a)) = ' ';
printf(' %c \n', c(1));
printf('%c', c(2:4)); printf('\n')
printf('%c', c(5:7)); printf('\n')
endfunction

