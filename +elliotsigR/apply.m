function a = apply(n,param)
	R = 0.1;
    rn = R * (rand(size(n))*2-1) .* n/1; 
a = n ./ (1 + abs(n))  + rn ;
end


