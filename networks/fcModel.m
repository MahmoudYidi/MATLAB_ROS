function dlY = fcModel(dlX, dlnet)
% Apply fully connected layer.

W = getDlnetVal(dlnet.Learnables,"fc","Weights");
B = getDlnetVal(dlnet.Learnables,"fc","Bias");
dlY = fullyconnect(dlX,W,B);

applySoftmax = getDlnetVal(dlnet.Operators,"fc","Softmax");
if (applySoftmax)
    dlY = softmax(dlY);
end

end