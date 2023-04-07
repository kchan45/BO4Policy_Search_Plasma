function Y = compute_objectives(w1,b1,wE,bE,Ns)
Nobj = 2;
Y=zeros(Ns,Nobj); %Matrix with all computed objectives per DNN
for k=1:Ns
%Here loop over samples and run CL trajectory
W1_test = py.numpy.array(w1(k,:,:));
b1_test = py.numpy.array(b1(k,:,:));
Wend_test = py.numpy.array(wE(k,:,:));
bend_test = py.numpy.array(bE(k,:,:));
[objs] = py.compute_objs_python.get_objs(W1_test,b1_test,Wend_test,bend_test,k);
objs = cell(objs);
obj1 = objs{1};
obj2 = objs{2};
Y(k,:) = [obj1,obj2];
end
end