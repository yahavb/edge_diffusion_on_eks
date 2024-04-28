aws iam create-policy --policy-name allow-access-to-model-assets --policy-document file://allow-access-to-model-assets.json

eksctl create iamserviceaccount --name appsimulator --namespace default --cluster tlvsummit-demo --role-name appsimulator \
    --attach-policy-arn arn:aws:iam::891377065549:policy/allow-access-to-model-assets --approve


create a role KedaOperatorRole-tlvsummit-demo

decode the role-arn
```
echo -n "arn:aws:iam::891377065549:role/KedaOperatorRole-tlvsummit-demo" | base64
```

create irsa with the role:

eksctl create iamserviceaccount \
  --cluster=tlvsummit-demo \
  --namespace=keda \
  --name=keda-operator \
  --role-name KedaOperatorRole-tlvsummit-demo \
  --attach-policy-arn=arn:aws:iam::aws:policy/AdministratorAccess \
  --approve

deploy the scaledobject
sd21-512-serve-scaledobject.yaml
