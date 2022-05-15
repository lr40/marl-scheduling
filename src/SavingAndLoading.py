import torch


def saveCheckpoint(PATH, world):
    topLayerDict = {}
    for i, agent in enumerate(world.agents):
        agentLayerDict = {}
        for j, targetNet in enumerate(agent.target["AcceptorNets"]):
            agentLayerDict["acceptorNet {}".format(j)] = targetNet.state_dict()
            agentLayerDict["acceptorNet {} optim".format(j)] = agent.optimizers[
                "acceptorOptimizers"
            ][j].state_dict()
        for j, targetNet in enumerate(agent.target["OfferNets"]):
            agentLayerDict["offerNet {}".format(j)] = targetNet.state_dict()
            agentLayerDict["offerNet {} optim".format(j)] = agent.optimizers["offerOptimizers"][
                j
            ].state_dict()
        topLayerDict["agentNets {}".format(i + 1)] = agentLayerDict
    torch.save(topLayerDict, PATH)


def loadCheckpoint(PATH, world):
    checkpointDict = torch.load(PATH)
    for i, agent in enumerate(world.agents):
        agentLayerDict = checkpointDict["agentNets {}".format(i + 1)]
        for j, policyNet in enumerate(agent.policy["AcceptorNets"]):
            policyNet.load_state_dict(agentLayerDict["acceptorNet {}".format(j)])
            policyNet.eval()
            agent.target["AcceptorNets"][j].load_state_dict(
                agentLayerDict["acceptorNet {}".format(j)]
            )
            agent.target["AcceptorNets"][j].eval()
            agent.optimizers["acceptorOptimizers"][j].load_state_dict(
                agentLayerDict["acceptorNet {} optim".format(j)]
            )
        for j, policyNet in enumerate(agent.policy["OfferNets"]):
            policyNet.load_state_dict(agentLayerDict["offerNet {}".format(j)])
            policyNet.eval()
            agent.target["OfferNets"][j].load_state_dict(agentLayerDict["offerNet {}".format(j)])
            agent.target["OfferNets"][j].eval()
            agent.optimizers["acceptorOptimizers"][j].load_state_dict(
                agentLayerDict["offerNet {} optim".format(j)]
            )
