(function () {
"use strict";

angular.module('public')
.controller('MenuController', MenuController);

MenuController.$inject = ['$state', '$q', 'helps', 'AuthService', 'HelpService', 'Web3HelpService'];
function MenuController($state, $q, helps, AuthService, HelpService, Web3HelpService) {
  var $ctrl = this;
  $ctrl.isAuthenticated = AuthService.isAuthenticated();
  $ctrl.currentUserProfileId = null;
  $ctrl.helps = helps || [];
  $ctrl.error = '';
  $ctrl.success = '';
  $ctrl.chainWarning = '';
  $ctrl.processing = false;
  $ctrl.walletBusy = false;
  $ctrl.walletError = '';
  $ctrl.appreciationDraft = {};
  $ctrl.showLedgerByHelp = {};
  $ctrl.loadingLedgerByHelp = {};
  $ctrl.ledgerEventsByHelp = {};
  $ctrl.ledgerErrorByHelp = {};
  $ctrl.ledgerDraftByHelp = {};
  $ctrl.walletSupported = Web3HelpService.isSupported();
  $ctrl.walletAddress = null;
  $ctrl.walletChainId = null;
  $ctrl.chainConfig = {
    contractAddress: '',
    preferredChainId: null
  };
  $ctrl.useChainActions = false;
  $ctrl.newHelp = defaultNewHelp();

  syncWalletState();

  $ctrl.loadHelps = function () {
    HelpService.getPrioritizedHelps()
      .then(function (results) {
        $ctrl.helps = results;
      })
      .catch(function (error) {
        $ctrl.error = (error.data && error.data.error) || 'Unable to load helps.';
      });
  };

  $ctrl.saveChainConfig = function () {
    clearMessages(false);
    Web3HelpService.setContractConfig({
      contractAddress: $ctrl.chainConfig.contractAddress,
      preferredChainId: $ctrl.chainConfig.preferredChainId
    });
    syncWalletState();
    $ctrl.success = 'Smart contract config saved.';
  };

  $ctrl.connectWallet = function () {
    clearMessages(false);
    $ctrl.walletBusy = true;

    Web3HelpService.setContractConfig({
      contractAddress: $ctrl.chainConfig.contractAddress,
      preferredChainId: $ctrl.chainConfig.preferredChainId
    });

    Web3HelpService.connectWallet()
      .then(function () {
        syncWalletState();
        $ctrl.success = 'Wallet connected.';
      })
      .catch(function (error) {
        $ctrl.walletError = parseWalletError(error);
      })
      .finally(function () {
        $ctrl.walletBusy = false;
      });
  };

  $ctrl.switchWalletChain = function () {
    clearMessages(false);
    $ctrl.walletBusy = true;

    Web3HelpService.setContractConfig({
      contractAddress: $ctrl.chainConfig.contractAddress,
      preferredChainId: $ctrl.chainConfig.preferredChainId
    });

    Web3HelpService.switchToPreferredChain()
      .then(function () {
        syncWalletState();
        $ctrl.success = 'Wallet switched to preferred chain.';
      })
      .catch(function (error) {
        $ctrl.walletError = parseWalletError(error);
      })
      .finally(function () {
        $ctrl.walletBusy = false;
      });
  };

  $ctrl.toggleChainActions = function () {
    if (!$ctrl.isAuthenticated) {
      $ctrl.useChainActions = false;
      Web3HelpService.setUseChainActions(false);
      return;
    }

    Web3HelpService.setUseChainActions(!!$ctrl.useChainActions);
    syncWalletState();
  };

  $ctrl.requireLogin = function () {
    $state.go('public.signin');
  };

  $ctrl.canOpenProfile = function (profileId) {
    return !!profileId &&
      profileId.indexOf('dummy-') !== 0 &&
      profileId.indexOf('mock-') !== 0 &&
      profileId.indexOf('self-') !== 0;
  };

  $ctrl.openProfile = function (profileId) {
    if (!$ctrl.canOpenProfile(profileId)) {
      return;
    }

    if ($ctrl.currentUserProfileId && profileId === $ctrl.currentUserProfileId) {
      $state.go('public.profile');
      return;
    }

    $state.go('public.userProfile', { profileId: profileId });
  };

  $ctrl.canStartChat = function (profileId) {
    return $ctrl.isAuthenticated &&
      $ctrl.canOpenProfile(profileId) &&
      (!$ctrl.currentUserProfileId || profileId !== $ctrl.currentUserProfileId);
  };

  $ctrl.isHelpParticipant = function (help) {
    if (!$ctrl.isAuthenticated || !$ctrl.currentUserProfileId || !help) {
      return false;
    }

    var requesterId = help.requester && help.requester.id;
    var helperId = help.helper && help.helper.id;
    return requesterId === $ctrl.currentUserProfileId || helperId === $ctrl.currentUserProfileId;
  };

  $ctrl.startChat = function (profileId) {
    if (!$ctrl.canStartChat(profileId)) {
      return;
    }

    $state.go('public.chat', { withProfileId: profileId });
  };

  $ctrl.createHelp = function () {
    clearMessages(true);
    $ctrl.processing = true;

    var payload = angular.copy($ctrl.newHelp);
    if (!$ctrl.isAuthenticated) {
      payload.help_type = 'request';
      payload.anonymous_requester = true;
      payload.requester_name = payload.requester_name || 'Anonymous Guest';
    }

    HelpService.createHelp(payload)
      .then(function (createdHelp) {
        return syncCreateOnChain(createdHelp, payload).then(function (chainMessage) {
          $ctrl.success = $ctrl.isAuthenticated ?
            'Help posted successfully.' :
            'Anonymous support request posted successfully.';
          if (chainMessage) {
            $ctrl.success += ' ' + chainMessage;
          }
          $ctrl.newHelp = defaultNewHelp();
          return $ctrl.loadHelps();
        });
      })
      .catch(function (error) {
        var apiErrors = error && error.data && error.data.errors;
        $ctrl.error = apiErrors ? apiErrors.join(', ') : 'Unable to post help.';
      })
      .finally(function () {
        $ctrl.processing = false;
      });
  };

  $ctrl.takeHelp = function (help) {
    runLifecycleAction({
      help: help,
      backendCall: function () { return HelpService.takeHelp(help.id); },
      chainAction: 'take',
      successMessage: 'Help taken successfully.',
      failureMessage: 'Unable to take help.'
    });
  };

  $ctrl.markInProgress = function (help) {
    runLifecycleAction({
      help: help,
      backendCall: function () { return HelpService.markInProgress(help.id); },
      chainAction: 'mark_in_progress',
      successMessage: 'Help marked in progress.',
      failureMessage: 'Unable to update help.'
    });
  };

  $ctrl.completeHelp = function (help) {
    runLifecycleAction({
      help: help,
      backendCall: function () { return HelpService.completeHelp(help.id); },
      chainAction: 'complete',
      successMessage: 'Help completed successfully.',
      failureMessage: 'Unable to complete help.'
    });
  };

  $ctrl.cancelHelp = function (help) {
    runLifecycleAction({
      help: help,
      backendCall: function () { return HelpService.cancelHelp(help.id); },
      chainAction: 'cancel',
      successMessage: 'Help cancelled.',
      failureMessage: 'Unable to cancel help.'
    });
  };

  $ctrl.sendAppreciation = function (help) {
    clearMessages(true);
    var draft = $ctrl.appreciationDraft[help.id] || {};
    var payload = {
      help_id: help.id,
      to_profile_id: draft.to_profile_id,
      reward_points: draft.reward_points || 1,
      note: draft.note || ''
    };

    HelpService.sendAppreciation(payload)
      .then(function () {
        $ctrl.success = 'Appreciation sent.';
        $ctrl.appreciationDraft[help.id] = {};
        return $ctrl.loadHelps();
      })
      .catch(function (error) {
        $ctrl.error = parseHttpError(error, 'Unable to send appreciation.');
      });
  };

  $ctrl.toggleLedger = function (help) {
    var helpId = help.id;
    $ctrl.showLedgerByHelp[helpId] = !$ctrl.showLedgerByHelp[helpId];
    if ($ctrl.showLedgerByHelp[helpId] && !$ctrl.ledgerEventsByHelp[helpId]) {
      $ctrl.loadLedger(help);
    }
    if ($ctrl.showLedgerByHelp[helpId] && !$ctrl.ledgerDraftByHelp[helpId]) {
      $ctrl.ledgerDraftByHelp[helpId] = {
        action: 'anchor',
        chain_state: 'submitted',
        chain_id: $ctrl.chainConfig.preferredChainId,
        contract_address: $ctrl.chainConfig.contractAddress,
        onchain_help_id: help.chain && help.chain.onchain_help_id
      };
    }
  };

  $ctrl.loadLedger = function (help) {
    var helpId = help.id;
    $ctrl.loadingLedgerByHelp[helpId] = true;
    $ctrl.ledgerErrorByHelp[helpId] = '';

    HelpService.getHelpLedger(helpId)
      .then(function (events) {
        $ctrl.ledgerEventsByHelp[helpId] = events;
      })
      .catch(function (error) {
        $ctrl.ledgerErrorByHelp[helpId] = parseHttpError(error, 'Unable to load audit log.');
      })
      .finally(function () {
        $ctrl.loadingLedgerByHelp[helpId] = false;
      });
  };

  $ctrl.recordChainEvent = function (help) {
    clearMessages(false);
    var helpId = help.id;
    var draft = $ctrl.ledgerDraftByHelp[helpId] || {};
    var payload = {
      action: draft.action || 'anchor',
      source_type: 'wallet',
      chain_state: draft.chain_state || 'submitted',
      tx_hash: draft.tx_hash,
      chain_id: draft.chain_id,
      contract_address: draft.contract_address,
      onchain_help_id: draft.onchain_help_id,
      metadata_hash: draft.metadata_hash,
      notes: draft.notes
    };

    if (payload.tx_hash) {
      payload.idempotency_key = buildIdempotencyKey(payload.chain_id, payload.tx_hash, payload.log_index || 0);
    }

    HelpService.recordHelpLedgerEvent(helpId, payload)
      .then(function () {
        $ctrl.success = 'Blockchain event recorded.';
        $ctrl.ledgerDraftByHelp[helpId] = {};
        $ctrl.loadLedger(help);
        return $ctrl.loadHelps();
      })
      .catch(function (error) {
        $ctrl.error = parseHttpError(error, 'Unable to record blockchain event.');
      });
  };

  $ctrl.shortHash = function (value) {
    if (!value || value.length < 16) {
      return value || 'n/a';
    }

    return value.slice(0, 10) + '...' + value.slice(-8);
  };

  if ($ctrl.isAuthenticated) {
    AuthService.getCurrentUser()
      .then(function (payload) {
        $ctrl.currentUserProfileId = payload && payload.profile && payload.profile.id;
      })
      .catch(function () {
        $ctrl.currentUserProfileId = null;
      });
  } else {
    $ctrl.useChainActions = false;
    Web3HelpService.setUseChainActions(false);
  }

  function runLifecycleAction(config) {
    clearMessages(true);

    config.backendCall()
      .then(function (updatedHelp) {
        return syncLifecycleOnChain(config.chainAction, updatedHelp || config.help).then(function (chainMessage) {
          $ctrl.success = config.successMessage;
          if (chainMessage) {
            $ctrl.success += ' ' + chainMessage;
          }
          return $ctrl.loadHelps();
        });
      })
      .catch(function (error) {
        $ctrl.error = parseHttpError(error, config.failureMessage);
      });
  }

  function syncCreateOnChain(createdHelp, createPayload) {
    if (!shouldUseChainActions() || !createdHelp || !createdHelp.id) {
      return $q.when(null);
    }

    return Web3HelpService.createHelpOnChain(createPayload)
      .then(function (txResult) {
        return recordWalletLedgerEvent(createdHelp.id, 'create', txResult, 'Created on smart contract')
          .then(function () {
            return 'On-chain create confirmed.';
          });
      })
      .catch(function (error) {
        appendChainWarning('On-chain create skipped: ' + parseWalletError(error));
        return null;
      });
  }

  function syncLifecycleOnChain(action, helpRecord) {
    if (!shouldUseChainActions()) {
      return $q.when(null);
    }

    if (!helpRecord || !helpRecord.id) {
      appendChainWarning('On-chain sync skipped: help context unavailable.');
      return $q.when(null);
    }

    var onchainHelpId = helpRecord.chain && helpRecord.chain.onchain_help_id;
    if (!onchainHelpId) {
      appendChainWarning('On-chain sync skipped: no on-chain help ID for this help.');
      return $q.when('On-chain sync pending.');
    }

    return Web3HelpService.runHelpActionOnChain(action, onchainHelpId)
      .then(function (txResult) {
        return recordWalletLedgerEvent(helpRecord.id, action, txResult, 'Lifecycle action synced from UI')
          .then(function () {
            return 'On-chain ' + humanizeAction(action) + ' confirmed.';
          });
      })
      .catch(function (error) {
        appendChainWarning('On-chain ' + humanizeAction(action) + ' failed: ' + parseWalletError(error));
        return 'On-chain sync pending.';
      });
  }

  function recordWalletLedgerEvent(helpId, action, txResult, notes) {
    var payload = {
      action: action,
      source_type: 'wallet',
      chain_state: txResult.chainState || 'confirmed',
      tx_hash: txResult.txHash,
      chain_id: txResult.chainId,
      contract_address: txResult.contractAddress,
      block_number: txResult.blockNumber,
      log_index: txResult.logIndex,
      wallet_address: txResult.walletAddress,
      metadata_hash: txResult.metadataHash,
      onchain_help_id: txResult.onchainHelpId,
      notes: notes
    };

    if (payload.tx_hash) {
      payload.idempotency_key = buildIdempotencyKey(payload.chain_id, payload.tx_hash, payload.log_index || 0);
    }

    return HelpService.recordHelpLedgerEvent(helpId, payload)
      .catch(function (error) {
        appendChainWarning('Ledger save failed for tx ' + $ctrl.shortHash(payload.tx_hash || '') + ': ' + parseHttpError(error, 'unknown backend error'));
      });
  }

  function buildIdempotencyKey(chainId, txHash, logIndex) {
    return [chainId || 'na', txHash || 'na', logIndex || 0].join(':');
  }

  function shouldUseChainActions() {
    return $ctrl.isAuthenticated && $ctrl.useChainActions && $ctrl.walletSupported;
  }

  function humanizeAction(action) {
    return (action || '').replace(/_/g, ' ');
  }

  function appendChainWarning(message) {
    if (!message) {
      return;
    }

    if ($ctrl.chainWarning) {
      $ctrl.chainWarning += ' ' + message;
      return;
    }

    $ctrl.chainWarning = message;
  }

  function parseWalletError(error) {
    if (!error) {
      return 'unknown wallet error';
    }
    if (error.message) {
      return error.message;
    }
    return String(error);
  }

  function parseHttpError(error, fallback) {
    return (error && error.data && (error.data.error || (error.data.errors && error.data.errors.join(', ')))) || fallback;
  }

  function clearMessages(clearWarning) {
    $ctrl.error = '';
    $ctrl.success = '';
    if (clearWarning) {
      $ctrl.chainWarning = '';
    }
    $ctrl.walletError = '';
  }

  function defaultNewHelp() {
    return {
      title: '',
      description: '',
      category: '',
      help_type: 'request',
      urgency: 3,
      scale: 3,
      required_resources: 3,
      time_available_hours: 24,
      supporters: 0,
      anonymous_requester: !$ctrl.isAuthenticated,
      requester_name: '',
      requester_contact: '',
      media_url: '',
      acknowledgement_url: ''
    };
  }

  function syncWalletState() {
    var chainState = Web3HelpService.getState();
    $ctrl.walletAddress = chainState.walletAddress;
    $ctrl.walletChainId = chainState.chainId;
    $ctrl.chainConfig.contractAddress = chainState.contractAddress || '';
    $ctrl.chainConfig.preferredChainId = chainState.preferredChainId;
    $ctrl.useChainActions = !!chainState.useChainActions;
  }
}

})();
