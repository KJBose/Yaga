(function () {
"use strict";

angular.module('common')
.service('Web3HelpService', Web3HelpService);

Web3HelpService.$inject = ['$window', '$q', '$injector'];
function Web3HelpService($window, $q, $injector) {
  var service = this;
  var STORAGE_KEY = 'yaga_chain_config_v1';
  var walletListenersBound = false;
  var defaultChainConfig = {
    defaultChainId: 11155111,
    contractAddress: '',
    useChainActionsByDefault: false
  };
  var chainConfig = resolveChainConfig();

  var state = {
    connected: false,
    walletAddress: null,
    chainId: null,
    preferredChainId: normalizeNumber(chainConfig.defaultChainId),
    contractAddress: normalizeAddress(chainConfig.contractAddress),
    useChainActions: !!chainConfig.useChainActionsByDefault
  };

  var HELP_REGISTRY_ABI = [
    'function createHelp(bytes32 metadataHash) payable returns (uint256 helpId)',
    'function offerHelp(uint256 helpId)',
    'function markInProgress(uint256 helpId)',
    'function completeHelp(uint256 helpId)',
    'function cancelHelp(uint256 helpId)',
    'function addReward(uint256 helpId) payable',
    'event HelpCreated(uint256 indexed helpId, address indexed requester, bytes32 metadataHash, uint256 escrowWei)',
    'event HelpMatched(uint256 indexed helpId, address indexed helper)',
    'event HelpInProgress(uint256 indexed helpId, address indexed actor)',
    'event HelpCompleted(uint256 indexed helpId, address indexed actor, address indexed helper, uint256 payoutWei)',
    'event HelpCancelled(uint256 indexed helpId, address indexed actor, uint256 refundedWei)',
    'event RewardAdded(uint256 indexed helpId, address indexed from, uint256 amountWei)'
  ];

  hydrateState();

  service.isSupported = function () {
    return !!($window.ethereum && $window.ethers && $window.ethers.providers && $window.ethers.Contract);
  };

  service.getState = function () {
    return angular.copy(state);
  };

  service.setContractConfig = function (config) {
    config = config || {};
    if (config.contractAddress !== undefined) {
      state.contractAddress = normalizeAddress(config.contractAddress);
    }
    if (config.preferredChainId !== undefined) {
      state.preferredChainId = normalizeNumber(config.preferredChainId);
    }
    persistState();
    return service.getState();
  };

  service.setUseChainActions = function (enabled) {
    state.useChainActions = !!enabled;
    persistState();
    return state.useChainActions;
  };

  service.connectWallet = function () {
    if (!service.isSupported()) {
      return $q.reject({ message: 'MetaMask/ethers not detected in browser.' });
    }

    bindWalletListeners();
    if (state.connected && state.walletAddress) {
      return refreshChainState()
        .then(function () {
          return ensurePreferredChain();
        })
        .then(function () {
          return service.getState();
        })
        .catch(function (error) {
          return $q.reject(normalizeError(error));
        });
    }

    return $q.when($window.ethereum.request({ method: 'eth_requestAccounts' }))
      .then(function (accounts) {
        if (!accounts || !accounts.length) {
          return $q.reject({ message: 'No wallet account returned.' });
        }

        state.walletAddress = accounts[0];
        state.connected = true;
        return refreshChainState();
      })
      .then(function () {
        return ensurePreferredChain().then(function () {
          return service.getState();
        });
      })
      .catch(function (error) {
        return $q.reject(normalizeError(error));
      });
  };

  service.switchToPreferredChain = function () {
    if (!service.isSupported()) {
      return $q.reject({ message: 'MetaMask/ethers not detected in browser.' });
    }

    if (!state.preferredChainId) {
      return $q.reject({ message: 'Set a preferred chain ID first.' });
    }

    var params = [{ chainId: toHexChainId(state.preferredChainId) }];
    return $q.when($window.ethereum.request({ method: 'wallet_switchEthereumChain', params: params }))
      .then(function () {
        return refreshChainState().then(function () {
          return service.getState();
        });
      })
      .catch(function (error) {
        if (error && error.code === 4902) {
          return $q.reject({ message: 'Selected chain is not added in wallet. Add it in MetaMask and retry.' });
        }
        return $q.reject(normalizeError(error));
      });
  };

  service.computeMetadataHash = function (helpPayload) {
    ensureEthersAvailable();
    var canonicalPayload = canonicalizeHelp(helpPayload || {});
    var json = JSON.stringify(canonicalPayload);
    return $window.ethers.utils.keccak256($window.ethers.utils.toUtf8Bytes(json));
  };

  service.createHelpOnChain = function (helpPayload, options) {
    options = options || {};
    var metadataHash = service.computeMetadataHash(helpPayload || {});

    return withContract(function (contract) {
      var overrides = {};
      if (options.escrowEth && Number(options.escrowEth) > 0) {
        overrides.value = $window.ethers.utils.parseEther(String(options.escrowEth));
      }
      return contract.createHelp(metadataHash, overrides).then(function (tx) {
        return tx.wait().then(function (receipt) {
          var eventInfo = extractEvent(receipt, 'HelpCreated');
          return buildTxResult({
            action: 'create',
            tx: tx,
            receipt: receipt,
            eventInfo: eventInfo,
            onchainHelpId: eventInfo && eventInfo.args && eventInfo.args.helpId ? Number(eventInfo.args.helpId.toString()) : null,
            metadataHash: metadataHash
          });
        });
      });
    });
  };

  service.runHelpActionOnChain = function (action, onchainHelpId, options) {
    options = options || {};
    var helpIdNumber = normalizeNumber(onchainHelpId);
    if (!helpIdNumber) {
      return $q.reject({ message: 'On-chain help ID is required.' });
    }

    return withContract(function (contract) {
      var txPromise;
      var eventName = null;

      if (action === 'take') {
        txPromise = contract.offerHelp(helpIdNumber);
        eventName = 'HelpMatched';
      } else if (action === 'mark_in_progress') {
        txPromise = contract.markInProgress(helpIdNumber);
        eventName = 'HelpInProgress';
      } else if (action === 'complete') {
        txPromise = contract.completeHelp(helpIdNumber);
        eventName = 'HelpCompleted';
      } else if (action === 'cancel') {
        txPromise = contract.cancelHelp(helpIdNumber);
        eventName = 'HelpCancelled';
      } else if (action === 'appreciation') {
        if (!options.rewardEth || Number(options.rewardEth) <= 0) {
          return $q.reject({ message: 'rewardEth is required for appreciation on-chain.' });
        }
        txPromise = contract.addReward(helpIdNumber, {
          value: $window.ethers.utils.parseEther(String(options.rewardEth))
        });
        eventName = 'RewardAdded';
      } else {
        return $q.reject({ message: 'Unsupported chain action: ' + action });
      }

      return txPromise.then(function (tx) {
        return tx.wait().then(function (receipt) {
          var eventInfo = extractEvent(receipt, eventName);
          return buildTxResult({
            action: action,
            tx: tx,
            receipt: receipt,
            eventInfo: eventInfo,
            onchainHelpId: helpIdNumber
          });
        });
      });
    });
  };

  function withContract(callback) {
    if (!service.isSupported()) {
      return $q.reject({ message: 'MetaMask/ethers not detected in browser.' });
    }

    if (!state.contractAddress) {
      return $q.reject({ message: 'Contract address is not configured.' });
    }

    return service.connectWallet()
      .then(function () {
        return ensurePreferredChain();
      })
      .then(function () {
        var provider = new $window.ethers.providers.Web3Provider($window.ethereum);
        var signer = provider.getSigner();
        var contract = new $window.ethers.Contract(state.contractAddress, HELP_REGISTRY_ABI, signer);
        return callback(contract);
      })
      .catch(function (error) {
        return $q.reject(normalizeError(error));
      });
  }

  function ensurePreferredChain() {
    if (!state.preferredChainId || !state.chainId || state.preferredChainId === state.chainId) {
      return $q.when(state.chainId);
    }
    return service.switchToPreferredChain().then(function () {
      return state.chainId;
    });
  }

  function refreshChainState() {
    if (!service.isSupported()) {
      return $q.when(null);
    }

    var provider = new $window.ethers.providers.Web3Provider($window.ethereum);
    return provider.getNetwork().then(function (network) {
      state.chainId = normalizeNumber(network.chainId);
      persistState();
      return state.chainId;
    });
  }

  function buildTxResult(params) {
    params = params || {};
    return {
      action: params.action,
      chainState: 'confirmed',
      txHash: params.tx && params.tx.hash,
      chainId: state.chainId,
      contractAddress: state.contractAddress,
      blockNumber: params.receipt && params.receipt.blockNumber,
      logIndex: params.eventInfo ? params.eventInfo.logIndex : null,
      onchainHelpId: params.onchainHelpId || null,
      metadataHash: params.metadataHash || null,
      walletAddress: state.walletAddress
    };
  }

  function extractEvent(receipt, eventName) {
    if (!receipt || !receipt.events || !receipt.events.length) {
      return null;
    }

    for (var i = 0; i < receipt.events.length; i++) {
      var event = receipt.events[i];
      if (event && event.event === eventName) {
        return event;
      }
    }

    return null;
  }

  function canonicalizeHelp(helpPayload) {
    return {
      title: asString(helpPayload.title),
      description: asString(helpPayload.description),
      category: asString(helpPayload.category),
      help_type: asString(helpPayload.help_type),
      urgency: normalizeNumber(helpPayload.urgency) || 0,
      scale: normalizeNumber(helpPayload.scale) || 0,
      required_resources: normalizeNumber(helpPayload.required_resources) || 0,
      time_available_hours: normalizeNumber(helpPayload.time_available_hours) || 0,
      supporters: normalizeNumber(helpPayload.supporters) || 0,
      media_url: asString(helpPayload.media_url),
      acknowledgement_url: asString(helpPayload.acknowledgement_url)
    };
  }

  function hydrateState() {
    var raw = $window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return;
    }

    try {
      var saved = JSON.parse(raw);
      state.preferredChainId = normalizeNumber(saved.preferredChainId) || state.preferredChainId;
      state.contractAddress = normalizeAddress(saved.contractAddress) || state.contractAddress;
      if (saved.useChainActions !== undefined) {
        state.useChainActions = !!saved.useChainActions;
      }
    } catch (e) {
      // ignore malformed local storage values
    }
  }

  function persistState() {
    $window.localStorage.setItem(STORAGE_KEY, JSON.stringify({
      preferredChainId: state.preferredChainId,
      contractAddress: state.contractAddress,
      useChainActions: state.useChainActions
    }));
  }

  function bindWalletListeners() {
    if (walletListenersBound || !$window.ethereum || !$window.ethereum.on) {
      return;
    }

    $window.ethereum.on('accountsChanged', function (accounts) {
      state.walletAddress = accounts && accounts.length ? accounts[0] : null;
      state.connected = !!state.walletAddress;
      if (!state.connected) {
        state.chainId = null;
      }
      persistState();
    });

    $window.ethereum.on('chainChanged', function () {
      refreshChainState();
    });

    walletListenersBound = true;
  }

  function normalizeError(error) {
    if (!error) {
      return { message: 'Unknown wallet error.' };
    }
    if (error.message) {
      return { message: error.message };
    }
    return { message: String(error) };
  }

  function toHexChainId(chainId) {
    return '0x' + Number(chainId).toString(16);
  }

  function asString(value) {
    return value === undefined || value === null ? '' : String(value).trim();
  }

  function normalizeNumber(value) {
    if (value === undefined || value === null || value === '') {
      return null;
    }
    var number = Number(value);
    return isNaN(number) ? null : number;
  }

  function normalizeAddress(value) {
    return asString(value);
  }

  function resolveChainConfig() {
    if (!$injector || !$injector.has || !$injector.has('ChainConfig')) {
      return defaultChainConfig;
    }

    try {
      return angular.extend({}, defaultChainConfig, $injector.get('ChainConfig') || {});
    } catch (e) {
      return defaultChainConfig;
    }
  }

  function ensureEthersAvailable() {
    if (!$window.ethers || !$window.ethers.utils) {
      throw new Error('ethers.js is not available in browser.');
    }
  }
}

})();
