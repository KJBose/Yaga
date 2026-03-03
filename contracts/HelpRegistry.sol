// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title HelpRegistry
/// @notice Minimal auditable help workflow contract for Yaga v1.
/// @dev Keep personal data off-chain. Store only metadata hashes and state.
contract HelpRegistry {
    enum HelpStatus {
        Open,
        Matched,
        InProgress,
        Completed,
        Cancelled
    }

    struct HelpRecord {
        address requester;
        address helper;
        bytes32 metadataHash;
        HelpStatus status;
        uint256 rewardEscrowWei;
        uint64 createdAt;
        uint64 updatedAt;
    }

    uint256 public nextHelpId = 1;
    mapping(uint256 => HelpRecord) public helps;

    event HelpCreated(uint256 indexed helpId, address indexed requester, bytes32 metadataHash, uint256 escrowWei);
    event HelpMatched(uint256 indexed helpId, address indexed helper);
    event HelpInProgress(uint256 indexed helpId, address indexed actor);
    event HelpCompleted(uint256 indexed helpId, address indexed actor, address indexed helper, uint256 payoutWei);
    event HelpCancelled(uint256 indexed helpId, address indexed actor, uint256 refundedWei);
    event RewardAdded(uint256 indexed helpId, address indexed from, uint256 amountWei);

    error HelpNotFound();
    error NotRequester();
    error NotParticipant();
    error InvalidState();
    error HelperUnavailable();
    error TransferFailed();

    function createHelp(bytes32 metadataHash) external payable returns (uint256 helpId) {
        helpId = nextHelpId++;
        helps[helpId] = HelpRecord({
            requester: msg.sender,
            helper: address(0),
            metadataHash: metadataHash,
            status: HelpStatus.Open,
            rewardEscrowWei: msg.value,
            createdAt: uint64(block.timestamp),
            updatedAt: uint64(block.timestamp)
        });

        emit HelpCreated(helpId, msg.sender, metadataHash, msg.value);
    }

    function offerHelp(uint256 helpId) external {
        HelpRecord storage record = _requireHelp(helpId);
        if (record.status != HelpStatus.Open) revert InvalidState();
        if (record.requester == msg.sender) revert HelperUnavailable();

        record.helper = msg.sender;
        record.status = HelpStatus.Matched;
        record.updatedAt = uint64(block.timestamp);

        emit HelpMatched(helpId, msg.sender);
    }

    function markInProgress(uint256 helpId) external {
        HelpRecord storage record = _requireHelp(helpId);
        if (!_isParticipant(record, msg.sender)) revert NotParticipant();
        if (record.status != HelpStatus.Matched) revert InvalidState();

        record.status = HelpStatus.InProgress;
        record.updatedAt = uint64(block.timestamp);

        emit HelpInProgress(helpId, msg.sender);
    }

    function completeHelp(uint256 helpId) external {
        HelpRecord storage record = _requireHelp(helpId);
        if (!_isParticipant(record, msg.sender)) revert NotParticipant();
        if (record.status != HelpStatus.InProgress && record.status != HelpStatus.Matched) revert InvalidState();
        if (record.helper == address(0)) revert HelperUnavailable();

        record.status = HelpStatus.Completed;
        record.updatedAt = uint64(block.timestamp);

        uint256 payout = record.rewardEscrowWei;
        record.rewardEscrowWei = 0;

        if (payout > 0) {
            (bool ok, ) = payable(record.helper).call{value: payout}("");
            if (!ok) revert TransferFailed();
        }

        emit HelpCompleted(helpId, msg.sender, record.helper, payout);
    }

    function cancelHelp(uint256 helpId) external {
        HelpRecord storage record = _requireHelp(helpId);
        if (record.requester != msg.sender) revert NotRequester();
        if (record.status == HelpStatus.Completed || record.status == HelpStatus.Cancelled) revert InvalidState();

        record.status = HelpStatus.Cancelled;
        record.updatedAt = uint64(block.timestamp);

        uint256 refund = record.rewardEscrowWei;
        record.rewardEscrowWei = 0;

        if (refund > 0) {
            (bool ok, ) = payable(record.requester).call{value: refund}("");
            if (!ok) revert TransferFailed();
        }

        emit HelpCancelled(helpId, msg.sender, refund);
    }

    function addReward(uint256 helpId) external payable {
        HelpRecord storage record = _requireHelp(helpId);
        if (record.status == HelpStatus.Cancelled) revert InvalidState();
        if (record.status == HelpStatus.Completed) revert InvalidState();
        if (msg.value == 0) revert InvalidState();

        record.rewardEscrowWei += msg.value;
        record.updatedAt = uint64(block.timestamp);

        emit RewardAdded(helpId, msg.sender, msg.value);
    }

    function getHelp(uint256 helpId)
        external
        view
        returns (
            address requester,
            address helper,
            bytes32 metadataHash,
            HelpStatus status,
            uint256 rewardEscrowWei,
            uint64 createdAt,
            uint64 updatedAt
        )
    {
        HelpRecord storage record = _requireHelp(helpId);
        return (
            record.requester,
            record.helper,
            record.metadataHash,
            record.status,
            record.rewardEscrowWei,
            record.createdAt,
            record.updatedAt
        );
    }

    function _requireHelp(uint256 helpId) internal view returns (HelpRecord storage record) {
        record = helps[helpId];
        if (record.requester == address(0)) revert HelpNotFound();
    }

    function _isParticipant(HelpRecord storage record, address actor) internal view returns (bool) {
        return actor == record.requester || actor == record.helper;
    }
}
