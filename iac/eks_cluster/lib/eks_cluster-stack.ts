import * as cdk from 'aws-cdk-lib';
import * as blueprints from '@aws-quickstart/eks-blueprints';
import { GlobalResources } from "@aws-quickstart/eks-blueprints";
import { VpcResourceProvider } from "./vpc_resource_provider";

import * as eks from "aws-cdk-lib/aws-eks";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import {UpdatePolicy} from "aws-cdk-lib/aws-autoscaling";
import {InstanceType} from "aws-cdk-lib/aws-ec2";
import {KubernetesVersion} from "aws-cdk-lib/aws-eks";

const version = 'auto';
const cluster_name = 'eks-edge-cluster';

export class EksClusterStack extends cdk.Stack {
  constructor(scope: cdk.App, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const account = this.account;
    const region = this.region;

    const edgeVPC = new VpcResourceProvider();

    const addOns: Array<blueprints.ClusterAddOn> = [
      new blueprints.addons.MetricsServerAddOn(),
      new blueprints.addons.ClusterAutoScalerAddOn(),
      new blueprints.addons.AwsLoadBalancerControllerAddOn(),
      new blueprints.addons.VpcCniAddOn(),
      new blueprints.addons.CoreDnsAddOn(),
      new blueprints.addons.KubeProxyAddOn()
    ];

    const clusterProvider = new blueprints.AsgClusterProvider(
        {
          id: "core",
          //vpc: edgeVPC.provide().vpcId),
          version: KubernetesVersion.of("auto"),
          desiredSize: 3,
          minSize: 1,
          maxSize: 3,
          //instanceType: new ec2.InstanceType('m5.large'),
          //vpcSubnets
          clusterName: `${cluster_name}`
        }
    )

      const clusterProvider2 = new blueprints.AsgClusterProvider(
          {
              id: "inference",
              //vpc: edgeVPC.provide().vpcId,
              version: KubernetesVersion.of("auto"),
              desiredSize: 3,
              minSize: 1,
              maxSize: 3,
              instanceType: new ec2.InstanceType('g5g.large'),
              //vpcSubnets
              clusterName: `${cluster_name}`
          }
      )

    const stack = blueprints.EksBlueprint.builder()
        .resourceProvider(GlobalResources.Vpc, edgeVPC)
        .clusterProvider(clusterProvider)
        .clusterProvider(clusterProvider2)
        .account(account)
        .region(region)
        .version(version)
        .addOns(...addOns)
        .useDefaultSecretEncryption(false) // set to false to turn secret encryption off (non-production/demo cases)
        .build(this, cluster_name);

  }
}